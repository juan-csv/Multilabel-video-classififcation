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


class YoutubeSegmentDataset(IterableDataset):
    def __init__(self, file_paths, seed=939, debug=False,
                 vocab_path="/Users/macbook/Desktop/OurGlass/VideoTagging/data/raw/yt8m_2nd/vocabulary.csv",
                 epochs=None, max_examples=None, offset=0):
        super(YoutubeSegmentDataset).__init__()
        print("Offset:", offset)
        self.file_paths = file_paths
        self.seed = seed
        self.debug = debug
        self.max_examples = max_examples
        vocab = pd.read_csv(vocab_path)
        self.label_mapping = {
            label: index for label, index in zip(vocab["Index"], vocab.index)
        }
        self.epochs = epochs
        self.offset = offset

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            seed = self.seed
        else:  # in a worker process
               # split workload
            if worker_info.num_workers > 1 and self.epochs == 1:
                raise ValueError("Validation cannot have num_workers > 1!")
            seed = self.seed + worker_info.id
        return self.generator(seed)
    
    def generator(self, seed):
        if self.epochs == 1:
            # validation
            tf_dataset = tf.data.TFRecordDataset(
                tf.data.Dataset.from_tensor_slices(self.file_paths)
            )
        else:
            tf_dataset = tf.data.TFRecordDataset(
                # tf.data.Dataset.list_files(
                #     "./data/train/*.tfrecord"
                # )
                tf.data.Dataset.from_tensor_slices(
                    self.file_paths
                ).shuffle(
                    100, seed=seed, reshuffle_each_iteration=True
                ).repeat(self.epochs)
            ).shuffle(256, seed=seed, reshuffle_each_iteration=True).repeat(self.epochs)
        for n_example, row in enumerate(self._iterate_through_dataset(tf_dataset)):
            if self.max_examples and self.max_examples == n_example:
                break
            yield row