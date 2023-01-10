import numpy as np
import json
import matplotlib.pyplot as plt

# Plot performance results

def plot_performance(folder, json_file):
  with open(folder + '/' + json_file + '.json', 'r') as file:
    results = json.load(file)

  s_lengths = []
  times = []
  sdrs = []


  for song in results:
    sample_lengths = results[song]
    for index, sample_length in enumerate(sample_lengths):
      if int(sample_length) not in s_lengths:
        s_lengths.append(int(sample_length))
        times.append([results[song][sample_length]["time"]])
        sdrs.append([results[song][sample_length]["sdr"]])
      else:
        times[index].append(results[song][sample_length]["time"])
        sdrs[index].append(results[song][sample_length]["sdr"])


  for index, s_length in enumerate(s_lengths):
    times[index] = np.mean(np.array(times[index]))
    sdrs[index] = np.mean(np.array(sdrs[index]))


  fig, ax1 = plt.subplots(figsize=(6, 2.5))

  ax2 = ax1.twinx()
  ax1.plot(s_lengths, sdrs, '-o', color='tab:blue')
  ax2.plot(s_lengths, times, '-o', color='tab:orange')

  ax1.set_xlabel('Sample length')
  ax1.set_ylabel('SDR values', color='tab:blue')
  ax2.set_ylabel('Time per prediction', color='tab:orange')

  ax1.grid()

  fig.savefig('images/' + json_file + '.png')

folder = 'results'
files = ['results_performance_vocals']

for file in files:
  plot_performance(folder, file)