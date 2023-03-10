"""
Inspired by a https://arxiv.org/pdf/1706.06905.pdf
"""

import yaml
import os, sys
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


class LinearResidualVideo(torch.nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        
        self.N_FEATURES_VIDEO = config['Dataset']['parameters']['N_FEATURES_VIDEO']
        self.N_CLASSES =        config['Dataset']['parameters']['N_CLASSES']
        
        self.emb = nn.Sequential(
                    nn.Linear(in_features=self.N_FEATURES_VIDEO, out_features=self.N_CLASSES),
                    nn.ReLU(),
                    #nn.BatchNorm1d(num_features=self.N_CLASSES)
                    )
        
        self.layer1 = nn.Sequential(
                    nn.Linear(in_features=self.N_CLASSES, out_features=self.N_CLASSES),
                    nn.ReLU())
        
        self.layer2 = nn.Sequential(
                    nn.Linear(in_features=self.N_CLASSES, out_features=self.N_CLASSES),
                    nn.ReLU())
        
        self.out = nn.Sequential(
                    nn.Linear(in_features=self.N_CLASSES, out_features=self.N_CLASSES),
                    nn.Sigmoid())
        
        # add dropout
        self.dropout = nn.Dropout(p=0.5)
        
        
    def forward(self, rgb_emb):
        """
            Args:
                rgb_emb: (batch, N_FEATURES_VIDEO)
            
        """
        # Embedding
        emb = self.emb(rgb_emb)
        emb = self.dropout(emb)
        
        # Residual
        res1 = self.layer1(emb)
        res1 = emb + res1
        res1 = self.dropout(res1)
        
        res2 = self.layer2(res1)
        res2 = res1 + res2
        res2 = self.dropout(res2)
        
        # OUT 
        out = self.out(res2)
        return out


class RNNModelVideo(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.N_FEATURES_VIDEO = config['Dataset']['parameters']['N_FEATURES_VIDEO']
        self.N_CLASSES =        config['Dataset']['parameters']['N_CLASSES']
        
        self.num_layer = 7
        self.hidden_size = 256
        
        self.lstm = nn.LSTM(input_size=self.N_FEATURES_VIDEO, hidden_size=self.hidden_size, num_layers=self.num_layer, batch_first=True, bidirectional=False)
        #self.rnn = nn.GRU(input_size=self.N_FEATURES_VIDEO, hidden_size=self.hidden_size, num_layers=self.num_layer, batch_first=True, bidirectional=False)
        
        self.middle = nn.Sequential(
                nn.Linear(in_features=self.hidden_size, out_features=self.N_CLASSES),
                nn.ReLU())
        
        self.out = nn.Sequential(
                nn.Linear(in_features=self.N_CLASSES, out_features=self.N_CLASSES),
                nn.Sigmoid())
        
        self.dropout = nn.Dropout(p=0.5)

        
    def forward(self, rgb_frames):
        """
        Args:
            rgb_frames: (batch, seq_len, N_FEATURES_VIDEO)
        """
        
        # in: (batch, seq_len, N_FEATURES_VIDEO) -> out: (batch, seq_len, hidden_size) hs[0]: (num_layers, batch, hidden_size)  hs[1]: (num_layers, batch, hidden_size) 
        out, hs = self.lstm(rgb_frames)
        out = self.dropout(out)
        
        # pooling, time dimension
        # in: (batch, seq_len, hidden_size) -> out: (batch, hidden_size)
        out = torch.mean(out, dim=1)
        out = self.dropout(out)
        
        # in: (batch, hidden_size) -> out: (batch, N_CLASSES)
        out = self.middle(out)
        out = self.dropout(out)
        
        # in: (batch, N_CLASSES) -> out: (batch, N_CLASSES)
        out = self.out(out)
        
        return out



class CNN1DSpatial(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.N_FEATURES_VIDEO = config['Dataset']['parameters']['N_FEATURES_VIDEO']
        self.N_CLASSES =        config['Dataset']['parameters']['N_CLASSES']
        
        self.linear_in = nn.Sequential(
                            nn.Linear(in_features=self.N_FEATURES_VIDEO, out_features=self.N_CLASSES)
                            )
        
        self.cnn1d = nn.Sequential(
                            nn.Conv1d(in_channels=self.N_CLASSES),
                            nn.BatchNorm1d(num_features=self.N_CLASSES),
                            nn.ReLU()
                            )
        
        self.cnn1d = nn.Sequential(
                            nn.Conv1d(in_channels=self.N_CLASSES),
                            nn.BatchNorm1d(num_features=self.N_CLASSES),
                            nn.ReLU()
                            )

        self.out = nn.Sequntial(
            nn.Linear(in_features=self.N_CLASSES, out_features=self.N_CLASSES),
            nn.Sigmoid()
        )
    
    def forward(self, rgb_emb):
        """
            rgb_emb: (batch, N_FEATURES_VIDEO)
        """
        out = self.linear_in(rgb_emb) # (batch, N_CLASSES)
        
        out = out.permute(0, 2, 1) # (batch, N_CLASSES, N_FEATURES_VIDEO)
        
        
        






if __name__ ==  "__main__":
    torch.backends.cudnn.enabled = False
    
    # Parameters
    BS = 5
    N_FEATURES_VIDEO = 1024
    N_FEATURES_AUDIO = 128
    N_CLASSES = 3862
    MAX_FRAMES = 360

    # create dummy input
    audio_data = np.random.randn( BS, N_FEATURES_AUDIO )
    video_data = np.random.randn( BS, N_FEATURES_VIDEO )
    frame_data = np.random.randn( BS, MAX_FRAMES, N_FEATURES_VIDEO )
    target = np.random.randint(0, 2, size=(BS, N_CLASSES))
    
    
    # get device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # convert to tensor
    audio_data = torch.from_numpy( audio_data ).float().to(device)
    video_data=  torch.from_numpy(video_data).float().to(device)
    frame_data = torch.from_numpy(frame_data).float().to(device)
    target = torch.from_numpy(target).float().to(device)
    
    lstm = RNNModelVideo(config).to(device)
    out = lstm(frame_data) # out: (batch, seq_len, hidden_size) | hs[0]: (num_layers, batch, hidden_size) | hs[1]: (num_layers, batch, hidden_size)
    print(f"Input shape:     {frame_data.shape}")
    print(f"Out shape:       {out.shape}")
    
    # define model
    model= LinearModelVideoAudio(config).to(device)
    out = model(video_data, audio_data)
    
    print(f"Input shape:     {audio_data.shape}")
    print(f"Out shape:       {out.shape}")

    #model= RNNModelVideo(config).to(device)
    #out = model(video_data)#, audio_data)
    
    print(f"Input shape:     {video_data.shape}")
    print(f"Out shape:       {out.shape}")