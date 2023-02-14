import os, sys
import yaml
import numpy as np
from pathlib import Path

# Define model
import torch 
from torch import nn
from torch.nn import Linear

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
with open(PATH_CONFIG) as f:
    config = yaml.safe_load(f)






class LinearModelVideoAudio(torch.nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        
        self.N_FEATURES_VIDEO = config['Dataset']['parameters']['N_FEATURES_VIDEO']
        self.N_FEATURES_AUDIO = config['Dataset']['parameters']['N_FEATURES_AUDIO']
        self.N_CLASSES =        config['Dataset']['parameters']['N_CLASSES']
        
        in_features = self.N_FEATURES_VIDEO + self.N_FEATURES_AUDIO
        # define layers
        
        # Embedding 
        self.emb = Linear(in_features=in_features, out_features=2024)
        
        # Activations
        self.activation = nn.ReLU()
        
        # Batchnorm
        self.batchnorm = nn.BatchNorm1d(num_features=2024)
        
        # out layer
        self.out = nn.Sequential(
            nn.Linear(in_features=2024, out_features=self.N_CLASSES),
            nn.Sigmoid()
        )
        
    def forward(self, video_emb, audio_emb):
        # concatenate embeddings
        x = torch.cat( [video_emb, audio_emb], axis=-1)
        emb = self.activation( self.emb(x) )
        emb = self.batchnorm( emb )
        out = self.out(emb)
        return out

class LinearModelVideo(torch.nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        
        self.N_FEATURES_VIDEO = config['Dataset']['parameters']['N_FEATURES_VIDEO']
        self.N_CLASSES =        config['Dataset']['parameters']['N_CLASSES']
        
        in_features = self.N_FEATURES_VIDEO
        # define layers
        
        # Embedding 
        self.emb = Linear(in_features=in_features, out_features=2024)
        
        # Activations
        self.activation = nn.ReLU()
        
        # Batchnorm
        self.batchnorm = nn.BatchNorm1d(num_features=2024)
        
        # out layer
        self.out = nn.Sequential(
            nn.Linear(in_features=2024, out_features=self.N_CLASSES),
            nn.Sigmoid()
        )
        
    def forward(self, video_emb):
        # concatenate embeddings
        emb = self.activation( self.emb(video_emb) )
        emb = self.batchnorm( emb )
        out = self.out(emb)
        return out


if __name__ ==  "__main__":
    # Parameters
    BS = 5
    N_FEATURES_VIDEO = 1024
    N_FEATURES_AUDIO = 128
    N_CLASSES = 3862

    # create dummy input
    audio_data = np.random.randn( BS, N_FEATURES_AUDIO )
    video_data = np.random.randn( BS, N_FEATURES_VIDEO )
    target = np.random.randint(0, 2, size=(BS, N_CLASSES))

    audio_data = torch.tensor( audio_data ).float()
    video_data=  torch.tensor(video_data).float()
    model= LinearModel(config)
    out = model(video_data, audio_data)
    
    print(f"Input shape:     {video_data.shape}")
    print(f"Out shape:       {out.shape}")