import glob
import yaml

import torch
import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow.compat.v1 as tf

from torch.utils.data import IterableDataset, DataLoader


# add root path
PATH_ROOT = Path.cwd()
for _ in range(6):
    last_files = os.listdir(PATH_ROOT)
    if 'src' in last_files:
        break
    else:
        PATH_ROOT = PATH_ROOT.parent
sys.path.append(PATH_ROOT.__str__())



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


# Read the data from multiple TFRecord files
dataset = tf.data.TFRecordDataset(list_train_files)


example = next(iter(dataset))


dataset = 
tf.io.parse_single_example(
    serialized=example,
    features={
        'mean_audio': tf.io.FixedLenFeature(shape=(128,), dtype=tf.float32),
        'mean_rgb': tf.io.FixedLenFeature(shape=(1024,), dtype=tf.float32),
        'labels': tf.io.VarLenFeature(dtype=tf.int64),
        'id': tf.io.FixedLenFeature(shape=(1,), dtype=tf.string)
    }
)
