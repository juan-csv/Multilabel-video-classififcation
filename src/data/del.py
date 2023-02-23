import torch
import os
import sys
import yaml
import logging
from pathlib import Path

import tensorflow as tf
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import pandas as pd
import glob



# add root path
PATH_ROOT = Path.cwd()
for _ in range(6):
    last_files = os.listdir(PATH_ROOT)
    if 'src' in last_files:
        break
    else:
        PATH_ROOT = PATH_ROOT.parent
sys.path.append(PATH_ROOT.__str__())

# load config
PATH_CONFIG = PATH_ROOT / 'config_files' / 'config.yaml'
with open( PATH_CONFIG ) as f:
    config = yaml.safe_load( f )
    
# Paths and parameters
PATH_VOCAB = PATH_ROOT / config['Dataset']['vocabulary_path']
N_CLASSES = config['Dataset']['parameters']['N_CLASSES']

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)

LOGGER = logging.getLogger("dataset")


class MyDataset(IterableDataset):
    def __init__(self,
                file_paths) -> None:
        super().__init__()
        
    def __iter__(self):
        # create generator
        pass
    
    def prepare_one_sample(self, row):
        example = tf.train.SequenceExample()
        tmp = example.FromString(row.numpy())
        context, features = tmp.context, tmp.feature_lists

        vid_labels = list(context.feature['labels'].int64_list.value)
        vid_labels_encoded = set([
            self.label_mapping[x] for x in vid_labels if x in self.label_mapping
        ])
        vid = context.feature['id'].bytes_list.value[0].decode('utf8')

        # Skip rows with empty labels for now
        if not vid_labels_encoded:
            # print("Skipped")
            return None, None

        # Expanded Lables: Shape (N_CLASSES)
        labels = np.zeros(N_CLASSES)
        labels[list(vid_labels_encoded)] = 1

        # Frames. Shape: (frames, 1024)
        pass
    
    def generator(self):
        pass



PATH_CONFIG = PATH_ROOT / 'config_files'
with open(PATH_CONFIG / 'config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Get relative dataset folder 
LEVEL_FEATURE = 'frame'

# Define paths
PATH_DATA = PATH_ROOT / config['Dataset']['folder'] / LEVEL_FEATURE
# List all tfrecords
train_pattern_files = PATH_DATA / 'train' / '*.tfrecord'  
list_train_files = glob.glob( train_pattern_files.__str__() )
filepaths = list_train_files[:3]
