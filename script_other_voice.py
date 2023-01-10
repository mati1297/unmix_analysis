import unmix_functions
import glob
import librosa
import numpy as np
from copy import copy
import json
import random


random.seed(500)

dataset_folder = 'dataset'
# Extracting
extract = "vocals"
logs = "vocal"

excluded_tracks = ["vocals", "bass", "drums", "other"]
excluded_tracks.remove(extract)

cant_per_song = 5

#  params
vqvae_step = '80001'
encoder_step = '120001'
file_des = 'trained'
sample_length = 147456
sample_freq = 44100

# Loading files of songs
folders_songs = sorted(glob.glob(dataset_folder + '/*'))
if 'dataset/Detsky Sad - Walkie Talkie' in folders_songs:
    folders_songs.remove('dataset/Detsky Sad - Walkie Talkie')

unmix_functions.initialize_mpi()

results = dict()

for song in folders_songs:
    length = int(librosa.get_duration(filename=song + '/mixture.wav', sr=sample_freq) * sample_freq // 1)
    results_song = dict()

    candidates = copy(folders_songs)
    candidates.remove(song)
    selected = random.sample(candidates, cant_per_song)
    for sel in selected:
        print(song, '+', sel)
        result_sel = dict()
        total_track = np.zeros((length), dtype=float)
        for t_ in excluded_tracks:
            x, _ = librosa.load(song + '/' + t_ + '.wav', sr=sample_freq, mono=True)
            total_track += x[0:length]
        other_track, _ = librosa.load(sel + '/' + extract + '.wav', sr=sample_freq, mono=True)
        if len(other_track) < length:
            other_track = np.pad(other_track, (0, length - len(other_track)))
        else:
            other_track = other_track[0:length]
        total_track += other_track
        
        try:
            prediction = unmix_functions.predict_channel(total_track, logs, vqvae_step, encoder_step, file_des, sample_length)
        except RuntimeError as e:
            print("Aqui ocurrio un error")
            continue

        results_song[sel] = unmix_functions.sdr(other_track, prediction)

        print("sdr = ", results_song[sel])

    results[song] = results_song

with open('results/results_other_' + extract + '.json', 'w') as file:
     file.write(json.dumps(results))
