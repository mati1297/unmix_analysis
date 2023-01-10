import glob
import librosa
import numpy as np
from copy import copy
import json
import matplotlib.pyplot as plt

song = 'dataset/Ben Carrigan - We\'ll Talk About It All Tonight'

sample_freq = 44100
up_to_freq = 10000

tracks = ['vocals', 'bass', 'drums', 'other']

titles = ['(a)', '(b)', '(c)', '(d)']

x, _ = librosa.load(song + '/mixture.wav', sr=sample_freq, mono=True)
length = len(x)

fft_size = int(up_to_freq / sample_freq * length / 2)
f = np.linspace(0, up_to_freq, fft_size)


results = np.zeros((4, fft_size))

fig, axs = plt.subplots(4, figsize=(6, len(tracks) * 2.5))

for index, track in enumerate(tracks):
    print(index)
    x, _ = librosa.load(song + '/' + track + '.wav', sr=sample_freq, mono=True)
    module = np.abs(np.fft.fft(x, n=length))
    results[index] = module[0:fft_size]

    axs[index].plot(f.flatten(), results[index, 0:fft_size], label=track)

    axs[index].text(0.97, 0.95, titles[index],
     horizontalalignment='right',
     verticalalignment='top',
     transform = axs[index].transAxes)

fig.tight_layout()
fig.savefig('images/fft.png')