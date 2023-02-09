import glob

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow.compat.v1 as tf

# add root path
PATH_ROOT = Path.cwd()
for _ in range(6):
    last_files = os.listdir(PATH_ROOT)
    if 'src' in last_files:
        break
    else:
        PATH_ROOT = PATH_ROOT.parent
sys.path.append(PATH_ROOT.__str__())