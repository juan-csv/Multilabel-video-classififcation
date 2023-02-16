import yaml
import os, sys
import requests
import pandas as pd
from dateutil import tz
from pathlib import Path
from datetime import datetime


# Add root path
PATH_ROOT = Path.cwd()
for _ in range(6):
    last_files = os.listdir(PATH_ROOT)
    if 'src' in last_files:
        break
    else:
        PATH_ROOT = PATH_ROOT.parent
sys.path.append(PATH_ROOT.__str__())

# Load config
PATH_CONFIG = PATH_ROOT / 'config_files' / 'config.yaml'
with open( PATH_CONFIG ) as f:
    config = yaml.safe_load( f )

def get_root_path():
    """
    For adding path
    sys.path.append(PATH_ROOT.__str__())
    """
    PATH_ROOT = Path.cwd()
    for _ in range(6):
        last_files = os.listdir(PATH_ROOT)
        if 'src' in last_files:
            break
        else:
            PATH_ROOT = PATH_ROOT.parent
    return PATH_ROOT

def current_tim2id():
    from_zone = tz.tzutc()
    to_zone = tz.tzlocal()
    utc = datetime.utcnow()
    utc = utc.replace(tzinfo=from_zone)
    local = utc.astimezone(to_zone)
    DATE_TIME = local.strftime('%m_%d_%H_%M')
    return DATE_TIME

class Map_index2label():
    def __init__(self):
        # get apth vocabulary
        PATH_VOCABULARY = PATH_ROOT / config['Dataset']['vocabulary_path']
        # TODO: load path from config file
        self.vocabulary_df = pd.read_csv(PATH_VOCABULARY)
        self.label_mapping = self.vocabulary_df[['Index', 'Name']].set_index('Index', drop=True).to_dict()['Name']

    def __call__(self, index):
        # check if index have len parameter
        if hasattr(index, '__len__'):
            return [self.label_mapping[i] for i in index]
        else:
            if index in list( self.label_mapping.keys() ):
                return self.label_mapping[index]
            else:
                return ''

def get_youtube_video_id(ID):
    url = f'http://data.yt8m.org/2/j/i/{ID[:2]}/{ID}.js'
    # get result of url

    response = requests.get(url).text

    response = response.replace('i(', '')
    response = response.replace(');', '')
    response = response.replace('"', '')
    VIDEO_ID = response.split(',')[-1]
    return VIDEO_ID, f"https://www.youtube.com/watch?v={VIDEO_ID}"