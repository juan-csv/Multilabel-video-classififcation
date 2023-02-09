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

# get N_SAMPLES
N_SAMPLES = 0
for tfrecord_file_path in list_train_files:
    try:
        for example in tf.python_io.tf_record_iterator(tfrecord_file_path):
            N_SAMPLES += 1
    except:
        print(tfrecord_file_path)

dataset_all = dataset.map( lambda x:
    tf.io.parse_single_example(
        serialized=x,
        features={
            'mean_audio': tf.io.FixedLenFeature(shape=(128,), dtype=tf.float32),
            'mean_rgb': tf.io.FixedLenFeature(shape=(1024,), dtype=tf.float32),
            'labels': tf.io.VarLenFeature(dtype=tf.int64),
            'id': tf.io.FixedLenFeature(shape=(1,), dtype=tf.string)
        }
    )
)


# 


class TFRecordDataset(torch.utils.data.Dataset):
    def __init__(self, tf_dataset):
        self.tf_dataset = tf_dataset

    def __getitem__(self, index):
        example = next(iter(self.tf_dataset.skip(index)))
        mean_audio = example['mean_audio'].numpy()
        mean_rgb = example['mean_rgb'].numpy()
        labels = example['labels'].values.numpy()
        id = example['id'].numpy()

        return torch.tensor(mean_audio)

    def __len__(self):
        return 10#len(self.tf_dataset)

pytorch_dataset = TFRecordDataset(dataset_all)
dataloader = torch.utils.data.DataLoader(pytorch_dataset, batch_size=32)
# take the second element of dataloader
data = next(iter(dataloader))


tfrecord_file_path = list_train_files[0]
for i, example in enumerate(tf.python_io.tf_record_iterator(tfrecord_file_path)):
    # get data for the video 
    tf_example = tf.train.SequenceExample.FromString(example)

    # Only for video level feature
    mean_audio_embedding_numpy = np.array(tf_example.context.feature['mean_audio'].float_list.value)    # shape: (128,)
    mean_rgb_embedding_numpy = np.array(tf_example.context.feature['mean_rgb'].float_list.value)      # shape: (1024,)

    # get index labels for the video
    labels =  tf_example.context.feature['labels'].int64_list.value # list of index
    
    # TODO: transform index to onehot target
    

    # get id video
    id = tf_example.context.feature['id'].bytes_list.value[0].decode(encoding='UTF-8') # str

    # get  audio and rgb embeddings
    audio_embedding_bytes =  tf_example.feature_lists.feature_list['audio'].feature # len() # number of frames
    rgb_embedding_bytes =  tf_example.feature_lists.feature_list['rgb'].feature     # len() # number of frames

    # get number of frames of video
    N_FRAMES_VIDEO = len(audio_embedding_bytes)

    # only take the embedding for the first frame
    if len(audio_embedding_bytes) == 0: # video level feature
        audio_embedding_numpy = mean_audio_embedding_numpy
        rgb_embedding_numpy = mean_rgb_embedding_numpy
    else: # Frame level feature
        audio_embedding_numpy = f_bytes2array( audio_embedding_bytes[0] ) # shape: (128,)
        rgb_embedding_numpy = f_bytes2array( rgb_embedding_bytes[0] ) # shape: (1024,)
    
    if i == 32:
        break
    
    print(np.abs(audio_embedding_numpy - data[i].numpy()).sum())



data.shape