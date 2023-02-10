import os
import pandas as pd
from dateutil import tz
from pathlib import Path
from datetime import datetime

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
    def __init__(self, PATH_VOCABULARY):
        # load vocabulary
        # TODO: load path from config file
        self.vocabulary_df = pd.read_csv(PATH_VOCABULARY)
        self.label_mapping = self.vocabulary_df[['Index', 'Name']].set_index('Index', drop=True).to_dict()['Name']

    def __call__(self, index):
        if index in list( self.label_mapping.keys() ):
            return self.label_mapping[index]
        else:
            return ''


