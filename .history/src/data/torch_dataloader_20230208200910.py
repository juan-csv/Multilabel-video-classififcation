import glob
import yaml

import os, sys
import numpy as np
from pathlib import Path
import tensorflow.compat.v1 as tf


# add root path
PATH_ROOT = Path.cwd()
for _ in range(6):
    last_files = os.listdir(PATH_ROOT)
    if 'src' in last_files:
        break
    else:
        PATH_ROOT = PATH_ROOT.parent
sys.path.append(PATH_ROOT.__str__())

# local imports


# Load config
PATH_CONFIG = PATH_ROOT / 'config_files'
with open(PATH_CONFIG / 'config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Get relative dataset folder 
LEVEL_FEATURE = config['Dataset']['feature_level']
FOLDER_LEVEL_FEATURE = 'frame_sample' if LEVEL_FEATURE=='frame' else 'video_sample'

# Define paths
PATH_DATA = PATH_ROOT / config['Dataset']['folder'] / FOLDER_LEVEL_FEATURE


# List all tfrecords
train_pattern_files = PATH_DATA / 'train*.tfrecord'
list_train_files = glob.glob( train_pattern_files.__str__() )

test_pattern_files = PATH_DATA / 'test*.tfrecord'
list_test_files = glob.glob( train_pattern_files.__str__() )



# Create function for reading tf records and extract features
x
        
        # TODO: Padding audio and rgb embedding

print(f"ID VIDEO:                       {id}")
print(f"Number of frames:               {N_FRAMES_VIDEO}")
print(f"Label index video:              {labels}")
#print(f"Names labels:                   {[map_index2label(index) for index in labels]}")
print(f"Shape video embedding:          {rgb_embedding_numpy.shape}")
print(f"Shape audio embedding:          {audio_embedding_numpy.shape}")
print(f"Shape mean video embedding:     {mean_rgb_embedding_numpy.shape}")
print(f"Shape mean adio embedding:      {mean_audio_embedding_numpy.shape}")
print('\n')

# create dataloader on pytorch for reading tfrecord 

import torch
from tfrecord.torch.dataset import TFRecordDataset

tfrecord_path = "/tmp/data.tfrecord"
index_path = None
description = None#{"image": "byte", "label": "float"}
dataset = TFRecordDataset(tfrecord_file_path, index_path, description)

loader = torch.utils.data.DataLoader(dataset, batch_size=2)

data = next(iter(loader))
print(data)