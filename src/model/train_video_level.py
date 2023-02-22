import os
import sys
import glob
import json
import yaml
import torch
import wandb
import random
import numpy as np
from torch import nn
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


# Add root path
PATH_ROOT = Path.cwd()
for _ in range(6):
    last_files = os.listdir(PATH_ROOT)
    if 'src' in last_files:
        break
    else:
        PATH_ROOT = PATH_ROOT.parent
sys.path.append(PATH_ROOT.__str__())

# Local imports
from utils.utils import current_tim2id
from src.data.torch_dataloader import DataManagerVideo
from src.data.torch_dataloader_v2 import YoutubeVideoDataset
from src.arquitectures.pt_models_video_level import LinearModelVideoAudio, LinearModelVideo, RNNModelVideo, LinearResidualVideo
from eval_util import calculate_gap, calculate_hit_at_one, calculate_precision_at_equal_recall_rate

# Functions
def get_subsample_files(list_files, PERCENTAGE, SHUFFLE=True):

    N_FILES = len(list_files)
    NUM_SAMPLES = int( N_FILES * PERCENTAGE )
    # Do shuffle
    if SHUFFLE:
        list_files = random.sample(list_files, NUM_SAMPLES)
    else:
        list_files = np.array(list_files)[:NUM_SAMPLES]
    
    return list_files
        

def instance_dataloaders2(PATH_DATA, config, PERCENTAGE=0.1):
    
    print("Creating train dataloader ...")
    train_pattern_files = PATH_DATA / 'train' / 'train*.tfrecord'
    list_train_files = glob.glob( train_pattern_files.__str__() )
    
    list_train_files, list_test_files = train_test_split(list_train_files, test_size=0.2, random_state=0)
    
    #list_train_files = get_subsample_files(list_train_files, PERCENTAGE, SHUFFLE=True)
    #list_test_files = get_subsample_files(list_test_files, PERCENTAGE , SHUFFLE=False)    
    
    #test_pattern_files = PATH_DATA / 'test' / '*.tfrecord'
    #list_test_files = glob.glob( test_pattern_files.__str__() )

    print(f"Number of train files: {len(list_train_files)}")
    print(f"Number of test files: {len(list_test_files)}")

    pytorch_dataset = YoutubeVideoDataset(list_train_files, epochs=1, USE_FEATURES=config['Dataset']['USE_FEATURES'])
    train_dataloader = DataLoader(pytorch_dataset, num_workers=0,
                        batch_size=config['Train']['bs'])#, collate_fn=collate_videos)
    
    pytorch_dataset = YoutubeVideoDataset(list_test_files, epochs=1, USE_FEATURES=config['Dataset']['USE_FEATURES'])
    test_dataloader = DataLoader(pytorch_dataset, num_workers=0,
                        batch_size=config['Train']['bs'])#, collate_fn=collate_videos)
    
    return train_dataloader, test_dataloader

# Load config
def load_config():
    PATH_CONFIG = PATH_ROOT / 'config_files' / 'config.yaml'
    with open( PATH_CONFIG ) as f:
        config = yaml.safe_load( f )
        
    return config



class TrainVideoTagging():
    def __init__(self, model, config, config_parameter_folders):
        self.config = config
        self.config_parameter_folders = config_parameter_folders
        self.PATH_DATA = config_parameter_folders['PATH_DATA'] 
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.EPOCHS = self.config['Train']['epochs']
        self.MODEL = self.config_parameter_folders['MODEL']
        self.PERCENTAGE_DATA = self.config['Train']['PERCENTAGE_DATA']
        self.USE_FEATURES = self.config['Dataset']['USE_FEATURES']
        # Set model to device
        self.model = model
        model.to(self.device)
        
        # Instance loss functions & metrics
        self.criterion = nn.BCELoss()

        # Instance optimizer
        self.optimizer = torch.optim.Adam( model.parameters() )

    def batch_forward(self, dataloader, MODE):
        # Define metrics
        loss_mean       = 0
        gap             = 0
        hit_at_one      = 0
        perr            = 0
        with tqdm(dataloader, desc=f"{MODE}ing [{self.MODEL}]...  Epoch {self.epoch+1}/{self.EPOCHS}", unit='batch') as pbar:
            for batch_index, features in enumerate(pbar):
                
                # Set features to device
                if 'audio' in self.USE_FEATURES:
                    video_emb,  audio_emb, target = features
                    video_emb,  audio_emb, target = video_emb.to(self.device).float(), audio_emb.to(self.device).float(), target.to(self.device).float()
                else:
                    video_emb, target = features
                    video_emb, target = video_emb.to(self.device).float(), target.to(self.device).float()
                
                # Reset gradient
                self.optimizer.zero_grad()
                
                # Get predictions
                if 'audio' in self.USE_FEATURES:
                    pred = self.model( video_emb, audio_emb )
                else:
                    pred = self.model( video_emb )
                
                # Get loss
                loss = self.criterion( pred, target )
                
                if MODE == 'train':
                    # Backward propagation
                    loss.backward()
                    # Update weights
                    self.optimizer.step()

                # Get metrics
                pred = pred.cpu().detach().numpy()
                target = target.cpu().detach().numpy()
                
                gap         += calculate_gap(pred, target)
                hit_at_one  += calculate_hit_at_one(pred, target)
                perr        += calculate_precision_at_equal_recall_rate(pred, target)
                
                loss_mean += loss.item()
                
                # Update tqdm
                pbar.set_postfix( 
                                    loss=loss_mean/(batch_index+1),
                                    gap=gap/(batch_index+1),
                                    hit_at_one=hit_at_one/(batch_index+1),
                                    perr=perr/(batch_index+1)
                )
                
                # Save model each 1000 steps
                if batch_index % 50e3 == 0:
                    self.batch_index = batch_index
                    #self.loss_mean = loss_mean/(batch_index+1)
                    #self.save_model()
                
        return {
            f'loss_{MODE}': loss_mean / (batch_index + 1),
            f'gap_{MODE}': gap / (batch_index + 1),
            f'hit_at_one_{MODE}': hit_at_one / (batch_index + 1),
            f'perr_{MODE}': perr / (batch_index + 1)
    }


    def train(self):
        print( f"training on [{self.device}]")
        print("---------------------------------\n")
        for epoch in range(self.EPOCHS):
            self.epoch = epoch
            self.model.train()
            
            # Use a percentage of the data selcted randomly
            if epoch == 0:
                train_dataloader, test_dataloader = instance_dataloaders2( self.PATH_DATA, self.config, self.PERCENTAGE_DATA )
            
            loss_train =  self.batch_forward(train_dataloader, MODE='train')

            #self.model.eval()
            with torch.no_grad():
                loss_test =  self.batch_forward(test_dataloader, MODE='test')

            # Instance wandb
            if self.epoch == 0:
                self.initialize_tracking_experiment()
                
            # Update metrics in wandb
            wandb.log( loss_test )
            wandb.log( loss_train )

            # Early stopping
            if self.early_stoping( loss_test['loss_test'] ):
                break
    
    def early_stoping(self, loss_test):
        if self.epoch > 0: 
            if loss_test < self.best_loss:
                # save and update 
                self.save_model(loss_test)
                self.best_loss = loss_test
                # reset counter
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter > self.config['Train']['patience']:
                    print(f"Early stopping at epoch --------> {self.epoch+1}")
                    return True
        
        else: # first epoch
            self.best_loss = loss_test
            self.patience_counter = 0
            self.save_model(loss_test)
            
        return False

    def save_model(self, loss_test):
        PATH_SAVE_MODEL = self.config_parameter_folders['PATH_SAVE_MODEL']
        NAME_EXPERIMENT = self.config_parameter_folders['NAME_EXPERIMENT']
        # save model
        if not os.path.exists( PATH_SAVE_MODEL ):
            os.makedirs( PATH_SAVE_MODEL )
            
        FULL_NAME = (PATH_SAVE_MODEL / f'{NAME_EXPERIMENT}_epoch_{self.epoch}_loss_test_{loss_test:.4f}').__str__()
        torch.save(self.model.state_dict(), f'{FULL_NAME}.pth')
        # save config as json
        with open(f'{FULL_NAME}.json', 'w') as f:
            json.dump(self.config, f)
        
        print(f'Model saved in --------------------> {FULL_NAME}')

    def initialize_tracking_experiment(self):
        # initalize wandb with token
        os.environ["WANDB_API_KEY"] = "01854ee351d4b1b98d7a61c0244450ce0125c163"

        wandb.init(
                    project = self.config_parameter_folders['NAME_PROJECT'], 
                    settings = wandb.Settings(start_method="fork"),
                    name = self.config_parameter_folders['NAME_EXPERIMENT'],
                    config = self.config['Train'],
                    resume = False,
                    tags = [self.config_parameter_folders['LEVEL_FEATURE'], self.config_parameter_folders['MODEL']],
                    notes = self.config_parameter_folders['NOTE'],
                    save_code = True
                    )

def main():
    config = load_config()
    # Get parameters

    # Arguments
    RUN_ID = None
    LEVEL_FEATURE = 'video'
    FEATURES = ['rgb'] # ['audio', 'rgb']
    NAME_EXPERIMENT = f"{RUN_ID}_baseline_{LEVEL_FEATURE}-level_{'_'.join(FEATURES)}"
    NAME_PROJECT = 'VideoTagging_YT8M_OurGlass'
    PERCENTAHE_DATA = 1
    MODEL = 'LinearResidualVideo'
    NOTE = 'Baseline'
    DIR_DATASET = None
    
    
    if RUN_ID == None:
        RUN_ID = current_tim2id()
        
    if DIR_DATASET == None:
        DIR_DATASET = config['Dataset']['folder']

    # Define Paths
    PATH_DATA =                 PATH_ROOT / DIR_DATASET / LEVEL_FEATURE
    FOLDERS_SAVE_MODEL =        PATH_ROOT / config['Model']['folder']
    PATH_SAVE_MODEL =           FOLDERS_SAVE_MODEL / LEVEL_FEATURE / MODEL / RUN_ID

    config_parameter_folders = {
        'RUN_ID': RUN_ID,
        'LEVEL_FEATURE': LEVEL_FEATURE,
        'NAME_EXPERIMENT': NAME_EXPERIMENT,
        'NAME_PROJECT': NAME_PROJECT,
        'MODEL': MODEL,
        'NOTE': NOTE,
        'PATH_DATA': PATH_DATA,
        'FOLDERS_SAVE_MODEL': FOLDERS_SAVE_MODEL,
        'PATH_SAVE_MODEL': PATH_SAVE_MODEL,
    }
    
    # Update config
    config['Dataset']['USE_FEATURES'] = FEATURES
    config['Dataset']['PATH_DATA'] = PATH_DATA.__str__()
    config['Train']['PERCENTAGE_DATA'] = PERCENTAHE_DATA
    config['Model']['model'] = MODEL
        
    print(f"Config;\n--------------------\n{config}\n")

    
    # Instance model
    if MODEL == 'LinearModel':
        model = LinearModelVideo( config )
    elif MODEL == 'RNNModelVideo':
        torch.backends.cudnn.enabled = False
        model = RNNModelVideo( config )
    elif MODEL == 'LinearResidualVideo':
        model = LinearResidualVideo( config )    
    
    print(f"Number parameters trainable:        {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"Number parameters total:            {sum(p.numel() for p in model.parameters())}\n")
    
    
    # TODO: Search checkpooint model


    # Instance training
    training_video_tagging = TrainVideoTagging(
                                            model,
                                            config,
                                            config_parameter_folders)
    
    # Training
    training_video_tagging.train()

    # TODO: Add callback for scheduler

main()