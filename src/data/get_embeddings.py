# Copyright 2017 Google Inc. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ref https://arxiv.org/pdf/1609.08675.pdf 
"""Facilitates extracting YouTube8M features from RGB images."""

import os, sys
from pathlib import  Path
# Add root path
PATH_ROOT = Path.cwd()
for _ in range(6):
    last_files = os.listdir(PATH_ROOT)
    if 'src' in last_files:
        break
    else:
        PATH_ROOT = PATH_ROOT.parent
sys.path.append(PATH_ROOT.__str__())

import tarfile
import numpy as np
from six.moves import urllib
#import tensorflow as tf
import tensorflow.compat.v1 as tf


INCEPTION_TF_GRAPH = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
YT8M_PCA_MAT = 'http://data.yt8m.org/yt8m_pca.tgz'
MODEL_DIR = os.path.join(os.getenv('HOME'), 'yt8m')


class YouTube8MFeatureExtractor(object):
    """Extracts YouTube8M features for RGB frames.
    First time constructing this class will create directory `yt8m` inside your
    home directory, and will download inception model (85 MB) and YouTube8M PCA
    matrix (15 MB). If you want to use another directory, then pass it to argument
    `model_dir` of constructor.
    If the model_dir exist and contains the necessary files, then files will be
    re-used without download.
    Usage Example:
        from PIL import Image
        import numpy
        # Instantiate extractor. Slow if called first time on your machine, as it
        # needs to download 100 MB.
        extractor = YouTube8MFeatureExtractor()
        image_file = os.path.join(extractor._model_dir, 'cropped_panda.jpg')
        im = numpy.array(Image.open(image_file))
        features = extractor.extract_rgb_frame_features(im)
    ** Note: OpenCV reverses the order of channels (i.e. orders channels as BGR
    instead of RGB). If you are using OpenCV, then you must do:
        im = im[:, :, ::-1]  # Reverses order on last (i.e. channel) dimension.
    then call `extractor.extract_rgb_frame_features(im)`
    """

    def __init__(self, model_dir=MODEL_DIR):
        # Create MODEL_DIR if not created.
        self._model_dir = model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Load PCA Matrix.
        download_path = self._maybe_download(YT8M_PCA_MAT)
        pca_mean = os.path.join(self._model_dir, 'mean.npy')
        if not os.path.exists(pca_mean):
            tarfile.open(download_path, 'r:gz').extractall(model_dir)
        self._load_pca()

        # Load Inception Network
        download_path = self._maybe_download(INCEPTION_TF_GRAPH)
        inception_proto_file = os.path.join(self._model_dir,
                                            'classify_image_graph_def.pb')
        if not os.path.exists(inception_proto_file):
            tarfile.open(download_path, 'r:gz').extractall(model_dir)
        self._load_inception(inception_proto_file)

    def extract_rgb_frame_features(self, frame_rgb, apply_pca=True):
        """Applies the YouTube8M feature extraction over an RGB frame.
        This passes `frame_rgb` to inception3 model, extracting hidden layer
        activations and passing it to the YouTube8M PCA transformation.
        Args:
          frame_rgb: numpy array of uint8 with shape (height, width, channels) where
            channels must be 3 (RGB), and height and weight can be anything, as the
            inception model will resize.
          apply_pca: If not set, PCA transformation will be skipped.
        Returns:
          Output of inception from `frame_rgb` (2048-D) and optionally passed into
          YouTube8M PCA transformation (1024-D).
        """
        assert len(frame_rgb.shape) == 3
        assert frame_rgb.shape[2] == 3  # 3 channels (R, G, B)
        with self._inception_graph.as_default():
            if apply_pca:
                frame_features = self.session.run(
                    'pca_final_feature:0', feed_dict={'DecodeJpeg:0': frame_rgb})
            else:
                frame_features = self.session.run(
                    'pool_3/_reshape:0', feed_dict={'DecodeJpeg:0': frame_rgb})
                frame_features = frame_features[0]
        return frame_features
    
    def extract_frame_level_features(self, frame_rgb, apply_pca=True):
        """

        Args:
            frame_rgb (np.array): shape (bs, n_frames, height, width, channels)
            apply_pca (bool, optional):  Defaults to True.

        Returns:
            frame_embeding_level (np.array): shape (bs, n_frames, 1024)
        """
        if len(frame_rgb.shape) == 4: # (n_frames, height, width, channels)
            frame_rgb = np.expand_dims(frame_rgb, axis=0)
        
        BS, N_FRAMES, _, _, _ = frame_rgb.shape
        N_FEATS_VIDEO = 1024
        
        # Instance output
        frame_embeding_level = np.zeros([BS, N_FRAMES, N_FEATS_VIDEO])
        
        # Iterate over batch
        for i_bs in range(BS):
            frame_batch = frame_rgb[i_bs]
            frame_embeding_level[i_bs] = np.array( list(map(lambda x: self.extract_rgb_frame_features(x, apply_pca=apply_pca), frame_batch)) )
                
        return frame_embeding_level
        
    
    def extract_video_level_features(self, frame_rgb, apply_pca=True):
        """

        Args:
            frame_rgb (np.array): shape (bs, n_frames, height, width, channels)
            apply_pca (bool, optional):  Defaults to True
        Returns:
            video_embeding_level (np.array): shape (bs, n_frames, 1024)
        """
        
        frame_embeding_level = self.extract_frame_level_features(frame_rgb, apply_pca=apply_pca) # (bs, n_frames, 1024)

        return frame_embeding_level

    def apply_pca(self, frame_features):
        """Applies the YouTube8M PCA Transformation over `frame_features`.
        Args:
          frame_features: numpy array of floats, 2048 dimensional vector.
        Returns:
          1024 dimensional vector as a numpy array.
        """
        # Subtract mean
        feats = frame_features - self.pca_mean

        # Multiply by eigenvectors.
        feats = feats.reshape((1, 2048)).dot(
            self.pca_eigenvecs).reshape((1024,))

        # Whiten
        feats /= np.sqrt(self.pca_eigenvals + 1e-4)
        return feats

    def _maybe_download(self, url):
        """Downloads `url` if not in `_model_dir`."""
        filename = os.path.basename(url)
        download_path = os.path.join(self._model_dir, filename)
        if os.path.exists(download_path):
            return download_path

        def _progress(count, block_size, total_size):
            sys.stdout.write(
                '\r>> Downloading %s %.1f%%' %
                (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        urllib.request.urlretrieve(url, download_path, _progress)
        statinfo = os.stat(download_path)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
        return download_path

    def _load_inception(self, proto_file):
        graph_def = tf.GraphDef.FromString(open(proto_file, 'rb').read())
        self._inception_graph = tf.Graph()
        with self._inception_graph.as_default():
            _ = tf.import_graph_def(graph_def, name='')
            self.session = tf.Session()
            Frame_Features = self.session.graph.get_tensor_by_name(
                'pool_3/_reshape:0')
            Pca_Mean = tf.constant(value=self.pca_mean, dtype=tf.float32)
            Pca_Eigenvecs = tf.constant(
                value=self.pca_eigenvecs, dtype=tf.float32)
            Pca_Eigenvals = tf.constant(
                value=self.pca_eigenvals, dtype=tf.float32)
            Feats = Frame_Features[0] - Pca_Mean
            Feats = tf.reshape(
                tf.matmul(tf.reshape(Feats, [1, 2048]), Pca_Eigenvecs), [
                    1024,
                ])
            tf.divide(Feats, tf.sqrt(Pca_Eigenvals + 1e-4),
                      name='pca_final_feature')

    def _load_pca(self):
        self.pca_mean = np.load(os.path.join(
            self._model_dir, 'mean.npy'))[:, 0]
        self.pca_eigenvals = np.load(
            os.path.join(self._model_dir, 'eigenvals.npy'))[:1024, 0]
        self.pca_eigenvecs = np.load(
            os.path.join(self._model_dir, 'eigenvecs.npy')).T[:, :1024]

if __name__ == "__main__":
    import numpy as np
    from src.data.get_youtube_video import get_video_frames, postporcess_video
    Frames, N_FRAMES, FPS = get_video_frames('hS5CfP8n_js')
    Frames = postporcess_video( Frames, N_FRAMES, FPS)
    
    # test
    extractor = YouTube8MFeatureExtractor()

    # get frame level representation
    frame_embeding_level = extractor.extract_frame_level_features(Frames)
    
    # get video level representation
    video_embeding_level = extractor.extract_video_level_features(Frames)



