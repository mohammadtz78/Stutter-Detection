# This code is under MIT licence with accompanying file LICENSE.md

"""
For each extracted clip:
* Get all clip information for that clip
* Save each clip's spectrogram as a new png file. 
"""

import argparse
import os
import pathlib
import pandas as pd
from scipy.io import wavfile
import pylab

parser = argparse.ArgumentParser(
    description='Extract spectrograms from clips.')
parser.add_argument('--labels', type=str,
                    help='Path to the labels csv files (e.g., SEP-28k_labels.csv)', default='data_prep/SEP-28k_labels.csv')
parser.add_argument('--clips', type=str,
                    help='Path where clips are stored.', default='data_prep/data')
parser.add_argument('--specs', type=str,
                    help='Path where spectrograms should be extracted.', default='data_prep/data')

args = parser.parse_args()
label_file = args.labels
data_dir = args.clips
output_dir = args.specs


# Load label/clip file
data = pd.read_csv(label_file, dtype={"EpId": str})

# Get label columns from data file
shows = data.Show
episodes = data.EpId
clip_idxs = data.ClipId
starts = data.Start
stops = data.Stop
labels = data.iloc[:, 5:].values

n_items = len(shows)

for i in range(n_items):
    clip_idx = clip_idxs[i]
    show_abrev = shows[i]
    episode = episodes[i].strip()

    # Setup paths
    clip_path = f"{data_dir}/clips/{show_abrev}/{episode}/{shows[i]}_{episode}_{clip_idx}.wav"
    spec_dir = pathlib.Path(f"{output_dir}/specs/{show_abrev}/{episode}/")
    spec_path = f"{spec_dir}/{shows[i]}_{episode}_{clip_idx}.png"

    # Check if spectrogram already exists
    if os.path.exists(spec_path):
        continue

    # Missing clip
    if not os.path.exists(clip_path):
        print("Missing", clip_path)
        continue

    # Verify spectrogram directory exists
    os.makedirs(spec_dir, exist_ok=True)

# Read file
    sample_rate, samples = wavfile.read(clip_path)

    # Generate spectrogram
    pylab.specgram(samples, Fs=sample_rate)

    # Save spectrogram to file
    pylab.savefig(f'{spec_path}')
    pylab.close()
