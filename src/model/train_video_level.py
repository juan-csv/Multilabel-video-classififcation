import os
import sys
import glob
import yaml
import torch
import wandb
import numpy as np
from torch import nn
from tqdm import tqdm
from pathlib import Path

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
from src.arquitectures.pt_models_video_level import LinearModel
from eval_util import calculate_gap, calculate_hit_at_one, calculate_precision_at_equal_recall_rate

# Functions
def instance_dataloaders(PATH_DATA, config):
    print("Creating train dataloader ...")
    train_pattern_files = PATH_DATA / 'train*.tfrecord'
    list_train_files = glob.glob( train_pattern_files.__str__() )

    test_pattern_files = PATH_DATA / 'test*.tfrecord'
    # TODO: change for test
    list_test_files = glob.glob( test_pattern_files.__str__() )

    pytorch_dataset = DataManagerVideo( list_train_files )
    train_dataloader = torch.utils.data.DataLoader(pytorch_dataset, batch_size=config['Train']['bs'], num_workers=0, shuffle=True)

    pytorch_dataset = DataManagerVideo( list_test_files )
    test_dataloader = torch.utils.data.DataLoader(pytorch_dataset, batch_size=config['Train']['bs'], num_workers=0)

    return train_dataloader, test_dataloader

# Load config
def load_config():
    PATH_CONFIG = PATH_ROOT / 'config_files' / 'config.yaml'
    with open( PATH_CONFIG ) as f:
        config = yaml.safe_load( f )
        
    print(f"Config;\n--------------------\n{config}\n")
    return config



class TrainVideoTagging():
    def __init__(self, model, config, config_parameter_folders):
        self.config = config
        self.config_parameter_folders = config_parameter_folders
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.EPOCHS = self.config['Train']['epochs']
        self.MODEL = self.config_parameter_folders['MODEL']
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
        with tqdm(dataloader, total=len(dataloader), desc=f"{MODE}ing [{self.MODEL}]...  Epoch {self.epoch+1}/{self.EPOCHS}", unit='batch') as pbar:
            for batch_index, features in enumerate(pbar):
                
                # Set features to device
                video_emb,  audio_emb, target = features
                video_emb,  audio_emb, target = video_emb.to(self.device), audio_emb.to(self.device), target.to(self.device)
                
                # Reset gradient
                self.optimizer.zero_grad()
                
                # Get predictions
                pred = self.model( video_emb, audio_emb )
                
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
                
        return {
            f'loss_{MODE}': loss_mean / batch_index + 1,
            f'gap_{MODE}': gap / batch_index + 1,
            f'hit_at_one_{MODE}': hit_at_one / batch_index + 1,
            f'perr_{MODE}': perr / batch_index + 1
    }


    def train(self, train_dataloader, test_dataloader):
        print( f"training on [{self.device}]")
        print("---------------------------------\n")
        for epoch in range(self.EPOCHS):
            self.epoch = epoch
            self.model.train()
            loss_train =  self.batch_forward(train_dataloader, MODE='train')

            self.model.eval()
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
                self.save_model()
                self.best_loss = loss_test
                # reset counter
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter > self.config['Train']['patience']:
                    print(f"Early stopping at epoch --------> {self.epoch+1}")
                    return True
        
        else: # first epoch
            self.best_loss = 0
            self.patience_counter = 0
            self.save_model()
            
        return False

    def save_model(self):
        PATH_SAVE_MODEL = self.config_parameter_folders['PATH_SAVE_MODEL']
        NAME_EXPERIMENT = self.config_parameter_folders['NAME_EXPERIMENT']
        # save model
        if not os.path.exists( PATH_SAVE_MODEL ):
            os.makedirs( PATH_SAVE_MODEL )
            
        full_PATH = (PATH_SAVE_MODEL / f'{NAME_EXPERIMENT}').__str__()
        torch.save(self.model.state_dict(), PATH_SAVE_MODEL / f'{NAME_EXPERIMENT}.pth')
        print(f"Model saved in --------------------> {full_PATH}")

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
    RUN_ID = current_tim2id()
    LEVEL_FEATURE = 'video'
    NAME_EXPERIMENT = f'{RUN_ID}_baseline_{LEVEL_FEATURE}-level'
    NAME_PROJECT = 'VideoTagging_YT8M'
    MODEL = 'LinearModel'
    NOTE = 'Baseline'

    # Define Paths
    FOLDER_LEVEL_FEATURE =      'frame_sample' if LEVEL_FEATURE=='frame' else 'video_sample'
    PATH_DATA =                 PATH_ROOT / config['Dataset']['folder'] / FOLDER_LEVEL_FEATURE
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
    
    # Instance model
    model = LinearModel( config )

    # TODO: Search checkpooint model


    # Instance dataloaders
    train_dataloader, test_dataloader = instance_dataloaders( PATH_DATA, config )

    # Instance training
    training_video_tagging = TrainVideoTagging(
                                            model,
                                            config,
                                            config_parameter_folders)
    
    # Training
    training_video_tagging.train(train_dataloader, test_dataloader)
    # TODO: Add callback for model checkpoint

    # TODO: Add callback for scheduler

main()