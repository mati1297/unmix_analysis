import numpy as np
import json
import matplotlib.pyplot as plt

# Plot alternative vocals results.

def format_combinations_text(text):
  text = text.replace('[\'', '')
  text = text.replace('\']', '')
  text = text.replace('\', \'', ' + ')
  return text

def plot_other(folder, json_file_original, json_file_double, json_file_other):
  to_plot = dict()
  
  to_plot['o (o)'] = []
  to_plot['a (a)'] = []
  to_plot['o (o + a)'] = []
  to_plot['a (o + a)'] = []
  to_plot['o + a (o + a)'] = []

  with open(folder + '/' + json_file_original + '.json', 'r') as file:
    results = json.load(file)

  for song in results: 
    for other in results[song]:
      to_plot['o (o)'].append(results[song]["['bass', 'drums', 'other', 'vocals']"])

  with open(folder + '/' + json_file_double + '.json', 'r') as file:
    results = json.load(file)

  for song in results:
    for other_song in results[song]:
      to_plot['o (o + a)'].append(results[song][other_song]['original'])
      to_plot['o + a (o + a)'].append(results[song][other_song]['both'])
      to_plot['a (o + a)'].append(results[song][other_song]['other'])

  with open(folder + '/' + json_file_other + '.json', 'r') as file:
    results = json.load(file)

  for song in results:
    for other_song in results[song]:
      to_plot['a (a)'].append(results[song][other_song])

  for key in to_plot:
    to_plot[key] = np.mean(np.array(to_plot[key]))


  fig, ax = plt.subplots(figsize=(6, 2.5))
  ax.set_title('SDR values for vocals combinations.')
  plt.bar(range(len(to_plot)), list(to_plot.values()), align='center', edgecolor='black', color=['tab:blue', 'tab:blue', 'tab:orange', 'tab:orange', 'tab:green'])
  plt.xticks(range(len(to_plot)), list(to_plot.keys()), rotation=30, ha='right')
  plt.ylabel('SDR values')

  rects = ax.patches

  for rect, label in zip(rects, to_plot.values()):
    label = np.round(label, 2)
    height = rect.get_height()
    ax.text(
        rect.get_x() + rect.get_width() / 2, height, label, ha="center", va="bottom"
    )

  plt.ylim([0, plt.ylim()[1] + 1])
  plt.tight_layout()
  plt.savefig('images/results_alternative_vocals.png')

folder = 'results'
files = [['results_comb_vocals', 'results_double_vocals', 'results_other_vocals']]

for file in files:
  plot_other(folder, file[0], file[1], file[2])