import glob
import yaml
import os, sys
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

# local imports

# Define paths
PATH_DATA = PATH_ROOT / 'data'
PATH_CONFIG = PATH_ROOT / 'config_files'

# Load config
with open(PA)

# List all tfrecords



# Create function for reading tf records and extract features
