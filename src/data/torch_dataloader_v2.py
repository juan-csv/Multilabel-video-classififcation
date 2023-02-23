"""
ref https://github.com/ceshine/yt8m-2019/blob/master/yt8m/dataloader.py 
"""

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


def Dequantize(feat_vector, max_quantized_value=2, min_quantized_value=-2):
    """Dequantize the feature from the byte format to the float format.
    Args:
      feat_vector: the input 1-d vector.
      max_quantized_value: the maximum of the quantized value.
      min_quantized_value: the minimum of the quantized value.
    Returns:
      A float vector which has the same shape as feat_vector.
    """
    assert max_quantized_value > min_quantized_value
    quantized_range = max_quantized_value - min_quantized_value
    scalar = quantized_range / 255.0
    bias = (quantized_range / 512.0) + min_quantized_value
    return feat_vector * scalar + bias


class YoutubeSegmentDataset(IterableDataset):
    def __init__(self, file_paths, debug=False,
                 vocab_path=PATH_VOCAB,
                 epochs=1, MAX_FRAMES=360,
                 USE_FEATURES= ['rgb', 'audio'] ):
        super(YoutubeSegmentDataset).__init__()
        self.file_paths = file_paths
        self.debug = debug
        self.MAX_FRAMES = MAX_FRAMES
        self.USE_FEATURES = USE_FEATURES
        vocab = pd.read_csv(vocab_path)
        self.label_mapping = {
            label: index for label, index in zip(vocab["Index"], vocab.index)
        }
        self.epochs = epochs

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        return self.generator()

    def prepare_one_sample(self, row):
        example = tf.train.SequenceExample()
        tmp = example.FromString(row.numpy())
        context, video_features = tmp.context, tmp.feature_lists

        vid_labels = list(context.feature['labels'].int64_list.value)
        vid_labels_encoded = set([
            self.label_mapping[x] for x in vid_labels if x in self.label_mapping
        ])
        
    # Skip rows with empty labels for now
        if not vid_labels_encoded:
            # print("Skipped")
            return None, None

        # Expanded Lables: Shape (N_CLASSES)
        labels = np.zeros(N_CLASSES)
        labels[list(vid_labels_encoded)] = 1
        """
        segment_labels = np.array(
            context.feature['segment_labels'].int64_list.value).astype("int64")
        segment_start_times = np.array(
            context.feature['segment_start_times'].int64_list.value)
        segment_scores = np.array(
            context.feature['segment_scores'].float_list.value).astype("int64")

        # Transform label
        segment_labels = np.array([
            self.label_mapping[x] for x in segment_labels])
        """
        vid = context.feature['id'].bytes_list.value[0].decode('utf8')


        # Frames. Shape: (frames, 1024)
        tmp = video_features.feature_list['rgb'].feature
        frames = tf.cast(tf.io.decode_raw(
            [x.bytes_list.value[0] for x in tmp], out_type="uint8"), "float32"
        ).numpy()

        # Audio. Shape: (frames, 128)
        tmp = video_features.feature_list['audio'].feature
        audio = tf.cast(tf.io.decode_raw(
            [x.bytes_list.value[0] for x in tmp], out_type="uint8"), "float32"
        ).numpy()

        # Pad agressively
        if frames.shape[0] < self.MAX_FRAMES:
            frames = np.concatenate([
                frames,
                np.zeros((self.MAX_FRAMES - frames.shape[0], frames.shape[1]))
            ])
        
        if audio.shape[0] < self.MAX_FRAMES:
            audio = np.concatenate([
                audio,
                np.zeros((self.MAX_FRAMES - audio.shape[0], audio.shape[1]))
            ])


        if 'audio' in self.USE_FEATURES:
            return (
                torch.from_numpy(frames),
                torch.from_numpy(audio),
                torch.from_numpy(labels)
            )
        else:
            return (
                torch.from_numpy(frames),
                torch.from_numpy(labels)
            )

    def _iterate_through_dataset(self, tf_dataset):
        for row in tf_dataset:
            if 'audio' in self.USE_FEATURES:
                frames, audio, labels = (
                    self.prepare_one_sample(row)
                )
                
                if frames is None:
                    continue
                
                yield frames, audio, labels
                
            else:
                frames, labels = (
                    self.prepare_one_sample(row)
                )
                
                if frames is None:
                    continue
                yield frames, labels

    def generator(self):
        tf_dataset = tf.data.TFRecordDataset(
            tf.data.Dataset.from_tensor_slices(self.file_paths)
        )

        for n_example, row in enumerate(self._iterate_through_dataset(tf_dataset)):
            yield row



class YoutubeVideoDataset(YoutubeSegmentDataset):
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
        frames = np.array(context.feature['mean_rgb'].float_list.value)

        # Audio. Shape: (frames, 128)
        audio = np.array(context.feature['mean_audio'].float_list.value)

        # Combine: shape(frames, 1152)
        features = torch.from_numpy(np.concatenate([frames, audio], axis=-1))

        if self.debug:
            print(f"http://data.yt8m.org/2/j/i/{vid[:2]}/{vid}.js")
            print(vid_labels_encoded)
            print(features.size(0))
            print("=" * 20 + "\n")

        if 'audio' in self.USE_FEATURES:
            return (
                torch.from_numpy(frames),
                torch.from_numpy(audio),
                torch.from_numpy(labels)
            )
        else:
            return (
                torch.from_numpy(frames),
                torch.from_numpy(labels)
            )
            

    def _iterate_through_dataset(self, tf_dataset):
        for row in tf_dataset:
            if 'audio' in self.USE_FEATURES:
                frames, audio, labels = (
                    self.prepare_one_sample(row)
                )
                
                if frames is None:
                    continue
                
                yield frames, audio, labels
            else:
                frames, labels = (
                    self.prepare_one_sample(row)
                )
                
                if frames is None:
                    continue
                yield frames, labels



def collate_videos(batch, pad=0):
    """Batch preparation.
    Pads the sequences
    """
    transposed = list(zip(*batch))
    max_len = max((len(x) for x in transposed[0]))
    data = torch.zeros(
        (len(batch), max_len, transposed[0][0].size(-1)),
        dtype=torch.float
    ) + pad
    masks = torch.zeros((len(batch), max_len), dtype=torch.float)
    for i, row in enumerate(transposed[0]):
        data[i, :len(row)] = row
        masks[i, :len(row)] = 1
    # Labels
    if transposed[1][0] is None:
        return data, masks, None
    labels = torch.stack(transposed[1]).float()
    # print(data.shape, masks.shape, labels.shape)
    return data, masks, labels


def collate_segments(batch, pad=0):
    """Batch preparation.
    Pads the sequences
    """
    #  frames, segment, (label, score, negative_mask)
    transposed = list(zip(*batch))
    max_len = max((len(x) for x in transposed[0]))
    video_data = torch.zeros(
        (len(batch), max_len, transposed[0][0].size(-1)),
        dtype=torch.float
    ) + pad
    video_masks = torch.zeros((len(batch), max_len), dtype=torch.float)
    for i, row in enumerate(transposed[0]):
        video_data[i, :len(row)] = row
        video_masks[i, :len(row)] = 1
    segments = torch.stack(transposed[1]).float()
    labels = torch.stack(transposed[2])
    return video_data, video_masks, segments, labels


def collate_test_segments(batch, pad=0, return_vid=True):
    """Batch preparation for the test dataset
    """
    #  video, segment, vid
    transposed = list(zip(*batch))
    max_len = max((len(x) for x in transposed[0]))
    video_features = torch.zeros(
        (len(batch), max_len, transposed[0][0].size(-1)),
        dtype=torch.float
    ) + pad
    video_masks = torch.zeros((len(batch), max_len), dtype=torch.float)
    for i, row in enumerate(transposed[0]):
        video_features[i, :len(row)] = row
        video_masks[i, :len(row)] = 1
    segment_features = torch.stack(transposed[1], dim=0)
    indices = transposed[2]
    if return_vid:
        vids = transposed[3]
        return video_features, video_masks, segment_features, indices, vids
    return video_features, video_masks, segment_features, indices



if __name__ == "__main__":
    import glob
    
    # Load config
    PATH_CONFIG = PATH_ROOT / 'config_files'
    with open(PATH_CONFIG / 'config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Get relative dataset folder 
    LEVEL_FEATURE = 'frame'

    # Define paths
    PATH_DATA = PATH_ROOT / config['Dataset']['folder'] / LEVEL_FEATURE
    # List all tfrecords
    train_pattern_files = PATH_DATA / 'train' / '*.tfrecord'    
    test_pattern_files = PATH_DATA / 'test' / '*.tfrecord'
    validate_pattern_files = PATH_DATA / 'validate' / '*.tfrecord'

    list_train_files = glob.glob( train_pattern_files.__str__() )
    list_test_files = glob.glob( test_pattern_files.__str__() )
    list_validate_files = glob.glob( validate_pattern_files.__str__() )
    
    filepaths = list_train_files[:3]
    

    USE_FEATURES = ['rgb', 'audio']  #['rgb'] or ['rgb', 'audio']
    
    dataset = YoutubeSegmentDataset(filepaths, epochs=1, USE_FEATURES=USE_FEATURES)
    loader = DataLoader(dataset, num_workers=0,
                        batch_size=1024, prefetch_factor=2)#, collate_fn=collate_videos)
    


    # iterate through loader
    
    for i, data in enumerate(loader):
        if i == 2:
            if 'audio' in USE_FEATURES:
                frames, audio, labels = data
                print(frames.size(), audio.size() ,labels.size())
            else:
                frames, labels = data
                print(frames.size(), labels.size())
            break


    # Counting number of batches
    from tqdm import tqdm
    NUM_BATCHES_TRAIN = 0

    with tqdm(loader, desc=f"counting...  ", unit='batch') as pbar:
        for batch_index, _ in enumerate(pbar):
            NUM_BATCHES_TRAIN += 1
            
    print(NUM_BATCHES_TRAIN, batch_index)