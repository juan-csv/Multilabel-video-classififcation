import os
import pandas as pd
from pathlib import Path


def get_root_path():
    """_summary_

    Returns:
        _type_: _description_
    """
    PATH_ROOT = Path.cwd()
    for _ in range(6):
        last_files = os.listdir(PATH_ROOT)
        if 'src' in last_files:
            break
        else:
            PATH_ROOT = PATH_ROOT.parent
    return PATH_ROOT

class Map_index2label():
    def __init__(self, PATH_VOCABULARY):
        # load vocabulary
        self.vocabulary_df = pd.read_csv(PATH_VOCABULARY)
        self.label_mapping = self.vocabulary_df[['Index', 'Name']].set_index('Index', drop=True).to_dict()['Name']

    def __call__(self, index):
        if index in list( self.label_mapping.keys() ):
            return self.label_mapping[index]
        else:
            return ''
