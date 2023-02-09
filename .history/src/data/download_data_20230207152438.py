import os
import sys
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

#cmd mkdir -p ~/data/yt8m/video; cd ~/data/yt8m/video

# Arguments
YOUTUBE_DATASET_VERSION = 2 # 1, 2 or 3
FRAME_LEVEL = False

LEVEL = 'frame_sample' if FRAME_LEVEL else 'video_sample'

# Define path to save
PATH_TO_SAVE = (PATH_ROOT / 'data' / 'raw' / f'yt8m_{YOUTUBE_DATASET_VERSION}_full' / LEVEL).__str__()

os.isdir
PATH_TO_SAVE