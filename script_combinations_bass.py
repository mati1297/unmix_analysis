import unmix_functions
import glob
import librosa
import numpy as np
from copy import copy
import json
    
dataset_folder = 'dataset'
# Extracting: 
extract = "bass"
logs = "bass"
# Present tracks
tracks = ["vocals", "other", "drums", "bass"] # Music file is vocals, unmix log file is vocal. Music file is other, in unmix is rest.
#unmix params
vqvae_step = '80001'
encoder_step = '120001'
file_des = 'trained'
sample_length = 147456
sample_freq = 44100


# Generating combinations
tracks.remove(extract)
combinations = [{extract}]

i = 0
while i < len(combinations): 
    for track in tracks:
        candidate = copy(combinations[i])
        candidate.add(track)
        if candidate not in combinations:
            combinations.append(candidate)
    i += 1

# Loading files of songs
folders_songs = sorted(glob.glob(dataset_folder + '/*'))
folders_songs = folders_songs

unmix_functions.initialize_mpi()

results = dict()

for song in folders_songs:
    length = int(librosa.get_duration(filename=song + '/mixture.wav', sr=sample_freq) * sample_freq // 1)
    print(song)
    target_track, _ = librosa.load(song + '/' + extract + '.wav', sr=sample_freq, mono=True)
    target_track = target_track[0:length]
    results_song = dict()
    for comb in combinations:
        total_track = np.zeros((length), dtype=float)
        print(comb)
        for track in comb: 
            x, _ = librosa.load(song + '/' + track + '.wav', sr=sample_freq, mono=True)
            x = x[0:length]
            total_track += x

        try:
            prediction = unmix_functions.predict_channel(total_track, logs, vqvae_step, encoder_step, file_des, sample_length)
        except RuntimeError as e:
            print("Aqui ocurrio un error")
            continue
            
        results_song[str(sorted(comb))] = unmix_functions.sdr(target_track, prediction)
        print("sdr = ", results_song[str(sorted(comb))])
    results[song] = results_song

with open('results/results_comb_' + extract + '.json', 'w') as file:
     file.write(json.dumps(results))
