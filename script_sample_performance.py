import unmix_functions
import glob
import librosa
import numpy as np
from copy import copy
import json
import random
import time

dataset_folder = 'dataset'
# Extracting
extract = "vocals"
logs = "vocal"

cant_per_song = 5

#unmix params
vqvae_step = '80001'
encoder_step = '120001'
file_des = 'trained'
sample_length = np.linspace(10000, 180000, 18, dtype=int)
sample_length += [49152, 98304, 147456, 196608]
sample_freq = 44100

sample_length = sorted(sample_length)

# Loading files of songs
folders_songs = sorted(glob.glob(dataset_folder + '/*'))
folders_songs = folders_songs

unmix_functions.initialize_mpi()

results = dict()

for song in folders_songs:
    print(song)
    length = int(librosa.get_duration(filename=song + '/mixture.wav', sr=sample_freq) * sample_freq // 1)

    target_track, _ = librosa.load(song + '/' + extract + '.wav', sr=sample_freq, mono=True)
    target_track = target_track[0:length]
    results_song = dict()

    x, _ = librosa.load(song + '/mixture.wav', sr=sample_freq, mono=True)

    for sample_length_ in sample_length:
        print(sample_length_)
        result_sample_length = dict()
        start = time.time()

        try:
            prediction = unmix_functions.predict_channel(x, logs, vqvae_step, encoder_step, file_des, int(sample_length_))
        except RuntimeError as e:
            print("Aqui ocurrio un error")
            continue

        result_sample_length['time'] = time.time() - start

        result_sample_length['sdr'] = unmix_functions.sdr(target_track, prediction)

        results_song[int(sample_length_)] = result_sample_length
        

    results[song] = results_song

with open('results/results_performance_' + extract + '.json', 'w') as file:
     file.write(json.dumps(results))
