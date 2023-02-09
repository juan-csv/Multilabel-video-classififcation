import os, sys
import yaml
import numpy as np
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

# load config
PATH_CONFIG = PATH_ROOT / 'config_files' / 'config.yaml'
with open(PATH_CONFIG) as f:
    config = yaml.safe_load(f)
config


# Define parameters
N_FEATURES_VIDEO = 1024
N_FEATURES_AUDIO = 128
N_CLASSES = 3862

# Define model
import torch 
from torch.nn import Linear

class LinearModel(torch.nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        
        self.N_FEATURES_VIDEO = config['Dataset']['parameters']['N_FEATURES_VIDEO']
        self.N_FEATURES_AUDIO = config['Dataset']['parameters']['N_FEATURES_AUDIO']
        self.N_CLASSES =        config['Dataset']['parameters']['N_CLASSES']
        
        in_features = self.N_FEATURES_VIDEO + self.N_FEATURES_AUDIO
        # define layers
        # Embedding
        linear = Linear(in_features=in_features, out_features=2024)
        