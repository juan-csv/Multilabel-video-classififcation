import glob
import yaml

import torch
import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
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


class DataManagerVideo(torch.utils.data.Dataset):
    def __init__(self, list_files, PATH_VOCABULARY= PATH_ROOT / 'data' / 'raw' / 'yt8m_2nd' / 'vocabulary.csv'):
        
        # get number of entites
        vocabulary_df = pd.read_csv(PATH_VOCABULARY)
        label_mapping = vocabulary_df[['Index', 'Name']].set_index('Index', drop=True).to_dict()['Name'] # dict: key --> index,  values --> name of category | len(): number of categories
        self.N_CLASSES = len(label_mapping)
        
        # Read the data from multiple TFRecord files
        dataset = tf.data.TFRecordDataset(list_files)
        
        # Prepare TFrecord 
        self.tf_dataset = dataset.map( lambda x:
            tf.io.parse_single_example(
                serialized=x,
                features={
                    'mean_audio': tf.io.FixedLenFeature(shape=(128,), dtype=tf.float32),
                    'mean_rgb': tf.io.FixedLenFeature(shape=(1024,), dtype=tf.float32),
                    'labels': tf.io.VarLenFeature(dtype=tf.int64)
                }
            )
        )
        
        self.N_SAMPLES = self.get_number_samples(list_files)


    def __getitem__(self, index):
        example = next(iter(self.tf_dataset.skip(index)))
        
        mean_audio = example['mean_audio'].numpy()
        mean_rgb = example['mean_rgb'].numpy()
        labels = example['labels'].values.numpy()
        # map labels index to sparse torch with shape self.N_CLASSES
        sparse_labels = torch.zeros(self.N_CLASSES)
        sparse_labels[labels] = 1

        return mean_audio, mean_rgb, sparse_labels
        
    def __len__(self):
        return self.N_SAMPLES
    
    def get_number_samples(self, list_files):
        # get N_SAMPLES
        N_SAMPLES = 0
        for tfrecord_file_path in list_files:
            for example in tf.data.TFRecordDataset(tfrecord_file_path):
                N_SAMPLES += 1
        return N_SAMPLES


if __name__ == '__main__':

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

    pytorch_dataset = DataManagerVideo( list_train_files )
    dataloader = torch.utils.data.DataLoader(pytorch_dataset, batch_size=4, num_workers=0)
    # take the second element of dataloader
    mean_audio, mean_rgb, labels = next(iter(dataloader))

    print(f" ")
