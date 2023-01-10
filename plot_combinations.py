import numpy as np
import json
import matplotlib.pyplot as plt

# Plot combinations results

def format_combinations_text(text, initials=True):
  if initials:
    text = text.replace('other', 'o')
    text = text.replace('bass', 'b')
    text = text.replace('drums', 'd')
    text = text.replace('vocals', 'v')
  text = text.replace('[\'', '')
  text = text.replace('\']', '')
  text = text.replace('\', \'', ' + ')
  return text

def plot_comb(folder, json_files):
  fig, axs = plt.subplots(len(json_files), figsize=(6, 2.5 * len(json_files)))

  titles = ['(a)', '(b)', '(c)', '(d)']
  
  max_ylim = 0

  for index, json_file in enumerate(json_files):
    with open(folder + '/' + json_file + '.json', 'r') as file:
      results = json.load(file)

    to_plot = dict()

    for song in results:
      combinations = results[song]
      for comb in combinations:
        new_text = format_combinations_text(comb)
        if new_text not in to_plot:
          to_plot[new_text] = [results[song][comb]]
        else:
          to_plot[new_text].append(results[song][comb])

    for key in to_plot:
      to_plot[key] = np.mean(np.array(to_plot[key]))


    axs[index].bar(range(len(to_plot)), list(to_plot.values()), align='center', edgecolor='black', color=['tab:blue', 'tab:orange', 'tab:orange', 'tab:orange', 'tab:green', 'tab:green', 'tab:green', 'tab:purple'])
    axs[index].set_xticks(range(len(to_plot)), list(to_plot.keys()), rotation=20, ha='right')
    axs[index].set_ylabel('SDR values')

    max_ylim = max(max_ylim, axs[index].get_ylim()[1])

    rects = axs[index].patches

    for rect, label in zip(rects, to_plot.values()):
      label = np.round(label, 2)
      height = rect.get_height()
      axs[index].text(
          rect.get_x() + rect.get_width() / 2, height, label, ha="center", va="bottom"
      )

    axs[index].text(0.97, 0.95, titles[index],
     horizontalalignment='right',
     verticalalignment='top',
     transform = axs[index].transAxes)

  for index, json_file in enumerate(json_files):
    axs[index].set_ylim([0, max_ylim + 1])

  fig.tight_layout()
  fig.savefig('images/results_combinations.png')

folder = 'results'
files = ['results_comb_vocals', 'results_comb_bass', 'results_comb_drums', 'results_comb_other']

plot_comb(folder, files)