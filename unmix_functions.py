
import time
import os
from unmix.vqvae.vqvae import VQVAE
from unmix.data.data_processor import DataProcessor
from unmix.utils.fp16 import FP16FusedAdam, FusedAdam, LossScalar, clipped_grad_scale, backward
from unmix.utils.ema import CPUEMA, FusedEMA, EMA
from unmix.utils.dist_utils import print_once, allreduce, allgather
from unmix.utils.torch_utils import zero_grad, count_parameters
from unmix.utils.audio_utils import audio_preprocess, audio_postprocess
from unmix.utils.logger import init_logging
from unmix.make_models import make_vqvae, restore_opt, save_checkpoint
from unmix.hparams import setup_hparams
from unmix.utils.remote_utils import download
from unmix.hparams import Hyperparams
from torch.nn.parallel import DistributedDataParallel
import unmix.utils.dist_adapter as dist
import torch as t
import numpy as np
import argparse
import warnings
import sys
from unmix.utils.dist_utils import setup_dist_from_mpi
import librosa

def sdr(references, estimates):
    # compute SDR for one song
    delta = 1e-7  # avoid numerical errors
    num = np.sum(np.square(references))
    den = np.sum(np.square(references - estimates[0:len(references)]))
    num += delta
    den += delta
    return 10 * np.log10(num / den)


def initialize_mpi(port=29500): 
    # for computing the module
    rank, local_rank, device = setup_dist_from_mpi(port=port)


def predict_channel(audio_vector, channel_name, vqvae_step, encoder_step, file_des, sample_length):

    class DefaultSTFTValues:
        def __init__(self, hps):
            self.sr = hps.sr
            self.n_fft = 2048
            self.hop_length = 256
            self.window_size = 6 * self.hop_length


    def calculate_bandwidth(hps):
        hps = DefaultSTFTValues(hps)

        bandwidth = dict(l2=1,
                        l1=1,
                        spec=1)
        return bandwidth

    t.cuda.empty_cache()

    REMOTE_PREFIX = 'https://openaipublic.azureedge.net/'
    #393216
    if file_des == "jukebox":
        print("Jukebox")
        hps = setup_hparams("vqvae", dict(name="vqvae_"+channel_name+"_predict", sr=44100, sample_length=sample_length, bs=4,
                                        labels=False, train=False, aug_shift=True, aug_blend=True, restore_vqvae=REMOTE_PREFIX + 'jukebox/models/5b/vqvae.pth.tar'))

    else:
        print("Trained")
        hps = setup_hparams("vqvae", dict(name="vqvae_"+channel_name+"_predict", sr=44100, sample_length=sample_length, bs=4,
                                        labels=False, train=False, aug_shift=True, aug_blend=True, restore_vqvae="/home/matias/Desktop/rrnn/unmix/logs/unmix_vqvae/vqvae_"+channel_name+"_b4/checkpoint_step_"+vqvae_step+".pth.tar"))

    encoder_cp = "/home/matias/Desktop/rrnn/unmix/logs/unmix_encoder/encoder_" + \
        channel_name+"_b4/checkpoint_step_"+encoder_step+".pth.tar"
    print(hps.restore_vqvae)
    print(encoder_cp)

    hps.ngpus = dist.get_world_size()

    print("gpus: ", hps.ngpus)

    block_kwargs = dict(width=hps.width, depth=hps.depth, m_conv=hps.m_conv,
                        dilation_growth_rate=hps.dilation_growth_rate,
                        dilation_cycle=hps.dilation_cycle,
                        reverse_decoder_dilation=hps.vqvae_reverse_decoder_dilation)

    vqvae = VQVAE(input_shape=(hps.sample_length, 1), levels=hps.levels, downs_t=hps.downs_t, strides_t=hps.strides_t,
                emb_width=hps.emb_width, l_bins=hps.l_bins,
                mu=hps.l_mu, commit=hps.commit,
                spectral=hps.spectral, multispectral=hps.multispectral,
                multipliers=hps.hvqvae_multipliers, use_bottleneck=hps.use_bottleneck,
                **block_kwargs)


    # restore from jukebox

    def load_checkpoint(path):
        restore = path
        if restore.startswith(REMOTE_PREFIX):
            remote_path = restore
            local_path = os.path.join(os.path.expanduser(
                "~/.cache"), remote_path[len(REMOTE_PREFIX):])
            if dist.get_rank() % 8 == 0:
                print("Downloading from azure")
                if not os.path.exists(os.path.dirname(local_path)):
                    os.makedirs(os.path.dirname(local_path))
                if not os.path.exists(local_path):
                    download(remote_path, local_path)
            restore = local_path
        dist.barrier()
        checkpoint = t.load(restore, map_location=t.device('cpu'))
        print("Restored from {}".format(restore))
        return checkpoint


    def restore_model(hps, model, checkpoint_path):
        model.step = 0
        if checkpoint_path != '':
            # checkpoint = t.load(checkpoint_path)
            checkpoint = load_checkpoint(checkpoint_path)
            checkpoint['model'] = {
                k[7:] if k[:7] == 'module.' else k: v for k, v in checkpoint['model'].items()}
            state_dict = checkpoint['model']
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    #print(name, ": ignored")
                    continue
                if isinstance(param, t.nn.parameter.Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                own_state[name].copy_(param)
                # if name == "bottleneck.level_blocks.0.k":
                #print(name, ": loaded")
                #    print(param)
            if 'step' in checkpoint:
                model.step = checkpoint['step']


    restore_model(hps, vqvae, hps.restore_vqvae)
    print("1. model restored")
    # load encoder
    checkpoint = t.load(encoder_cp)
    checkpoint['model'] = {
        k[7:] if k[:7] == 'module.' else k: v for k, v in checkpoint['model'].items()}

    state_dict = checkpoint['model']
    own_state = vqvae.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            #print(name, ": ignored")
            continue
        if isinstance(param, t.nn.parameter.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)
        #print(name, ": loaded")
    print("2. model restored")



    def load_prompts(audio_vector, duration, offset, hps):
        xs = []
        x = audio_vector[np.newaxis, offset:offset+duration if duration else None]
        x = x.T  # CT -> TC

        # while len(xs) < hps.sample_length:
        #    xs.extend(xs)
        #xs = xs[:hps.sample_length]
        #x = t.stack([t.from_numpy(x) for x in xs])
        x = t.from_numpy(x)
        x = t.stack([x])
        x = x.to('cuda', non_blocking=True)
        return x

    # from unmix.utils.dist_utils import setup_dist_from_mpi
    # rank, local_rank, device = setup_dist_from_mpi(port=29500)
    # hps.ngpus = dist.get_world_size()


    def get_ddp(model):
        rank = dist.get_rank()
        local_rank = rank % 8
        # ddp = DistributedDataParallel(model, device_ids=[
        #                              local_rank], output_device=local_rank, broadcast_buffers=False, bucket_cap_mb=hps.bucket)

        ddp = t.nn.DataParallel(model)
        ddp.to("cuda")
        print("Number of gpus:")
        print(t.cuda.device_count())
        return ddp


    #vqvae = get_ddp(vqvae)
    vqvae.eval()
    vqvae.to('cuda', non_blocking=True)
    hps.bandwidth = calculate_bandwidth(hps)
    forw_kwargs = dict(loss_fn=hps.loss_fn, hps=hps)
    #duration = (hps.sample_length/hps.sr)

    #print("param: ", list(vqvae.parameters())[0][0])

    #for ind, audio_file in enumerate(audio_files):
        # to predict the first model: uncomment or for unmix comment
        # audio_file = audio_file.replace("_0", channel_id)

    song_samples = len(audio_vector)
    #print("file: ", ind)

    chunks = int(song_samples//hps.sample_length)
    offset = 0

    x_total = np.array([])

    for j in range(0, chunks):
        x = load_prompts(audio_vector, hps.sample_length, offset, hps)
        offset += hps.sample_length
        start_time = time.time()
        x_out, loss, _metrics = vqvae(x, **forw_kwargs)
        #print("duration: ", duration)
        #print("time: ", time.time()-start_time)

        x_out = x_out[0].cpu().detach().numpy()

        x_out = x_out.reshape(x_out.shape[0]).flatten()
        x_out = x_out.reshape(x_out.shape[0], 1)

        #audio_name = audio_file.split('/')[-1]
        #np.save(save_dir + "/"+file_des+"_" +
        #        audio_name+"_"+str(j).zfill(3)+".npy", x_out)
        x_total = np.concatenate((x_total, x_out.flatten()))


    # last chunk

    x = load_prompts(audio_vector, None, offset, hps)

    start_time = time.time()
    x_out, loss, _metrics = vqvae(x, **forw_kwargs)

    x_out = x_out[0].cpu().detach().numpy()

    x_out = x_out.reshape(x_out.shape[0]).flatten()
    x_out = x_out.reshape(x_out.shape[0], 1)

    #audio_name = audio_file.split('/')[-1]
    #np.save(save_dir+"/"+file_des+"_" +
    #        audio_name+"_"+str(chunks).zfill(3)+".npy", x_out)
    x_total = np.concatenate((x_total, x_out.flatten()))

    return x_total
