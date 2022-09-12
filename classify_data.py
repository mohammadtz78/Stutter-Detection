# This code is under MIT licence with accompanying file LICENSE.md

"""
For each extracted spectrogram:
* Get all clip information for that spectrogram
* Classify and save each spectrogram as a new png file. 
"""

import argparse
import os
import pathlib
import shutil
import pandas as pd

parser = argparse.ArgumentParser(description='Classify spectrograms.')
parser.add_argument('--labels', type=str,
                    help='Path to the labels csv files (e.g., SEP-28k_labels.csv)', default='data_prep/SEP-28k_labels.csv')
parser.add_argument('--specs', type=str,
                    help='Path where spectrograms are stored.', default='data_prep/data')
parser.add_argument('--dataset', type=str,
                    help='Path where to save the dataset.', default='dataset')

args = parser.parse_args()
label_file = args.labels
data_dir = args.specs
output_dir = args.dataset


# Load label/clip file
data = pd.read_csv(label_file, dtype={"EpId": str})

# Get label columns from data file
shows = data.Show
episodes = data.EpId
clip_idxs = data.ClipId
starts = data.Start
stops = data.Stop
labels = data.iloc[:, 5:].values
porlongations = data.Prolongation
blocks = data.Block
soundreps = data.SoundRep
wordreps = data.WordRep
intejections = data.Interjection


n_items = len(shows)

loaded_clip = ""
for i in range(n_items):
    clip_idx = clip_idxs[i]
    show_abrev = shows[i]
    episode = episodes[i].strip()
    stutters = True if (porlongations[i] != 0 or blocks[i] != 0 or soundreps[i]
                        != 0 or wordreps[i] != 0 or intejections[i] != 0) else False

    # Setup paths
    spec_path = f"{data_dir}/specs/{show_abrev}/{episode}/{shows[i]}_{episode}_{clip_idx}.png"
    if (stutters):
        dataset_class = "Class_0"
    else:
        dataset_class = "Class_1"
    dataset_dir = pathlib.Path(f"{output_dir}/{dataset_class}/")
    dataset_path = f"{dataset_dir}/{shows[i]}_{episode}_{clip_idx}.png"
    if not os.path.exists(spec_path):
        print("Missing", spec_path)
        continue

    # Verify dataset directory exists
    os.makedirs(dataset_dir, exist_ok=True)

    # Copy to the class directory
    shutil.copy(spec_path, dataset_path)
