import glob
import yaml
import os, sys
from pathlib import Path

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
import tensorflow.compat.v1 as tf
tfrecord_file = list_train_files[0]
for example in tf.python_io.tf_record_iterator(PATH_TF):
    # get data for the video 
    tf_example = tf.train.SequenceExample.FromString(example)

    # Only for video level feature
    mean_audio_embedding_numpy = np.array(tf_example.context.feature['mean_audio'].float_list.value)    # shape: (128,)
    mean_rgb_embedding_numpy = np.array(tf_example.context.feature['mean_rgb'].float_list.value)      # shape: (1024,)

    # get index labels for the video
    labels =  tf_example.context.feature['labels'].int64_list.value # list of index

    # get id video
    id = tf_example.context.feature['id'].bytes_list.value[0].decode(encoding='UTF-8') # str

    # get  audio and rgb embeddings
    audio_embedding_bytes =  tf_example.feature_lists.feature_list['audio'].feature # len() # number of frames
    rgb_embedding_bytes =  tf_example.feature_lists.feature_list['rgb'].feature     # len() # number of frames

    # get number of frames of video
    N_FRAMES_VIDEO = len(audio_embedding_bytes)

    # only take the embedding for the first frame
    if len(audio_embedding_bytes) == 0:
        audio_embedding_numpy = mean_audio_embedding_numpy
        rgb_embedding_numpy = mean_rgb_embedding_numpy
    else:
        audio_embedding_numpy = f_bytes2array( audio_embedding_bytes[0] ) # shape: (128,)
        rgb_embedding_numpy = f_bytes2array( rgb_embedding_bytes[0] ) # shape: (1024,)

