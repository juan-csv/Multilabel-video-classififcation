"""

"""

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

import yaml
import torch
import numpy as np

# Local imports
from src.arquitectures.pt_models_video_level import LinearModelVideoAudio, LinearModelVideo
from src.data.get_embeddings import YouTube8MFeatureExtractor
from src.data.get_youtube_video import get_video_frames, postporcess_video
from utils.utils import Map_index2label

PATH_CONFIG = PATH_ROOT / 'config_files' / 'config.yaml'
with open( PATH_CONFIG ) as f:
    config = yaml.safe_load( f )






def get_path_model(FOLDERS_SAVE_MODEL, RUN_ID):
    PATH_SAVE_MODEL = FOLDERS_SAVE_MODEL / LEVEL_FEATURE / MODEL / RUN_ID
    # load model
    list_files = [f.split('.pth')[0] for f in os.listdir(PATH_SAVE_MODEL) if f.endswith('.pth')]

    # get the batch of each file
    batch_list = [int(f.split('_batch_')[-1].split('_')[0]) for f in list_files]
    last_batch_idx = np.argsort(batch_list)[-1]

    PATH_MODEL = PATH_SAVE_MODEL / list_files[last_batch_idx]
    return PATH_MODEL


class ModelTag():
    def __init__(self, PATH_MODEL):
        self.PATH_MODEL = PATH_MODEL
        
        # Get device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Load model and config
        self.load_model_and_config()
        
        # Instance extractor
        self.extractor = YouTube8MFeatureExtractor()

        # Instance label2index
        self.index2label = Map_index2label()

        
    def load_model_and_config(self):
        
        PATH_MODEL = self.PATH_MODEL.__str__()
        
        # load config
        PATH_CONFIG = PATH_MODEL + '.json'
        with open(PATH_CONFIG) as f:
            config = yaml.safe_load(f)
        
        # load model
        #if config['Model']['parameters']['model'] == 'LinearModelVideoAudio':
            #model = LinearModelVideoAudio( config )
        #elif config['Model']['parameters']['model'] == 'LinearModelVideo':
        self.model = LinearModelVideo( config )
        self.model.load_state_dict(torch.load(PATH_MODEL + '.pth' ))
            
        # Turn to inference mode
        self.model.eval()
        
        # move model to device
        self.model.to(self.device)
        
    
    def __call__(self, Frames, min_prob=0.8):
        """_summary_

        Args:
            Frames (np.array): (BS ,N_samples, width, height, channels)  or (N_samples, width, height, channels)
            min_prob (float, optional): [min confidence value]. Defaults to 0.8.
        Returns:
            label_pred (np.array): categorie predicted
            confidence (np.array): confidence of the prediction
        """
        
        if Frames.ndim == 4: # (N_samples, width, height, channels) (no batch size)
            Frames =  Frames[np.newaxis,:] # (1, N_samples, width, height, channels) 
            pass
        
        # get video level representation
        video_embeding_level = self.extractor.extract_video_level_features(Frames) # (BS, 1024)
        
        # Get label prediction
        in_feats = torch.tensor(video_embeding_level).float().to(self.device)   # (BS, 1024)
        predictions = self.model(torch.tensor(in_feats))    # (BS, N_CLASSES)

        # Format predictions
        label_pred, confidence = self.format_predictions(predictions, min_prob)
        
        return label_pred, confidence


    def format_predictions(self, predictions, min_prob):
        """_summary_

        Args:
            predictions (torch.tensor): (BS, N_CLASSES)
        Returns:
            label_pred (np.array): categorie predicted
            confidence (np.array): confidence of the prediction
        """
        # Iterate for each batch
        label_pred = []
        confidence = []
        for pred in predictions:
            # Format predictions
            index_pred = np.argsort(pred.cpu().detach().numpy())[::-1]
            score_pred = np.sort(pred.cpu().detach().numpy())[::-1]

            # Index to labels
            labels = np.array( self.index2label(index_pred) ) # (N_CLASSES, 1)

            # filter by min_prob
            label_pred.append( list( labels[score_pred > min_prob]) )
            confidence.append( list( score_pred[score_pred > min_prob]) )
        
        return label_pred, confidence
        

if __name__ == '__main__':

    Testing = [('eguZ69v_vlQ', ['Toy', 'Littlest Pet Shop']), ('ER9Hdp04tWs', ['Association football', 'Stadium']), ('ETF2-Zz3J18', ['Light']), ('jtvbLq9bYRc', ['Circle']), ('6BPXQMxdHog', ['Vehicle', 'Car', 'Brake']), ('-j989rqetQE', ['Musician', 'Guitar', 'String instrument', 'Electric guitar', 'Epiphone']), ('F-4h2WwVr3g', ['Game', 'Cartoon', 'Animation']), ('UZt7rP0poxs', ['Game', 'Association football', 'Highlight film']), ("<?xml version='1.0' encoding='UTF-8'?><Error><Code>AccessDenied</Code><Message>Access denied.</Message></Error>", ['Game', 'Video game', 'Fighting game', 'Street Fighter', 'Super Street Fighter IV']), ('kKZBuy8kaj8', ['Cartoon', 'Animation']), ('kLVJJvEQN44', ['Harry Potter']), ('GLqhiVWGm8Q', ['Concert']), ('-QUa6WgjDqc', ['Game', 'Video game', 'Drum kit', 'Rock Band', 'Rock Band (video game)'])]



    # Instance Model
    RUN_ID = '02_14_10_15'
    LEVEL_FEATURE = 'video'
    FEATURES = ['rgb'] # ['audio', 'rgb']
    NAME_EXPERIMENT = f"{RUN_ID}_baseline_{LEVEL_FEATURE}-level_{'_'.join(FEATURES)}"
    MODEL = 'LinearModel'
    FOLDERS_SAVE_MODEL =        PATH_ROOT / config['Model']['folder']
    PATH_MODEL = get_path_model(FOLDERS_SAVE_MODEL, RUN_ID)
    # Load model
    model_video = ModelTag(PATH_MODEL)


    # Do predictions
    for Id in range(len(Testing)):
        try:
            youtube_video_id, target = Testing[Id][0], Testing[Id][1]
            # get video
            Frames, N_FRAMES, FPS = get_video_frames(youtube_video_id)
            Frames = postporcess_video( Frames, N_FRAMES, FPS)
            label_pred, score_pred = model_video(Frames)

            list(zip(label_pred, score_pred))

            #print(target,'\n',label_pred, '\n')
            print(f"Target:    {target} \nPredicted: {label_pred} \nUrl: https://www.youtube.com/watch?v={youtube_video_id} \n---------------------------------------")
        except:
            print(f"Error: {youtube_video_id}")
        