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

# local imports



class YoutubeVideoDataset(IterableDataset):
    def __init__(self, file_paths, seed=939, debug=False,
                 vocab_path="/Users/macbook/Desktop/OurGlass/VideoTagging/data/raw/yt8m_2nd/vocabulary.csv",
                 epochs=None, max_examples=None, offset=0):
        super(YoutubeVideoDataset).__init__()
        print("Offset:", offset)
        self.file_paths = file_paths
        self.seed = seed
        self.debug = debug
        self.max_examples = max_examples
        vocab = pd.read_csv(vocab_path)
        self.label_mapping = {
            label: index for label, index in zip(vocab["Index"], vocab.index)
        }
        self.N_CLASSES = vocab['Index'].unique()
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

        # Expanded Lables: Shape (1000)
        labels = np.zeros(self.N_CLASSES, dtype=np.int)
        labels[list(vid_labels_encoded)] = 1

        # Frames. Shape: (frames, 1024)
        frames = np.array( context.feature['mean_rgb'].float_list.value )

        # Audio. Shape: (frames, 128)
        audio = np.array( context.feature['mean_audio'].float_list.value )

        # Combine: shape(frames, 1152)
        features = torch.from_numpy(np.concatenate([frames, audio], axis=-1))

        if self.debug:
            print(f"http://data.yt8m.org/2/j/i/{vid[:2]}/{vid}.js")
            print(vid_labels_encoded)
            print(features.size(0))
            print("=" * 20 + "\n")

        return (
            features,
            # (1000,)
            torch.from_numpy(labels)
        )

    def _iterate_through_dataset(self, tf_dataset):
        for row in tf_dataset:
            features, labels = (
                self.prepare_one_sample(row)
            )
            if features is None:
                continue
            yield features, labels



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

            
print(list_train_files[:2])
dataset = YoutubeVideoDataset(list_train_files, ep)
loader = DataLoader(dataset, num_workers=0, batch_size=2)
max_class = 0
for i, (segments, label) in enumerate(loader):
    max_class = max(max_class, label[0, :].max())
    break
    print(segments.size(), label.size(), label[0])
    print(max_class)