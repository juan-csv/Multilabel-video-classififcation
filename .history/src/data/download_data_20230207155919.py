""" Downloading data from youtube8M dataset
https://research.google.com/youtube8m/download.html


For runining script:

python src/data/download_data.py \
        --youtube_dataset_version 2 \
        --no-frame_level
"""

import os
import sys
import argparse
from pathlib import Path

# define root path
# add root path
PATH_ROOT = Path.cwd()
for _ in range(6):
    last_files = os.listdir(PATH_ROOT)
    if 'src' in last_files:
        break
    else:
        PATH_ROOT = PATH_ROOT.parent
sys.path.append(PATH_ROOT.__str__())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--youtube_dataset_version', type=int, default=2)
    parser.add_argument('--frame_level', dest='frame_level', action='store_true')
    parser.add_argument('--no-frame_level', dest='frame_level', action='store_false')
    
    # Set default values for boolean variables
    parser.set_defaults(frame_level=True)
    
    args = parser.parse_args()
    
    # Arguments
    YOUTUBE_DATASET_VERSION = args.youtube_dataset_version # 1, 2 or 3
    FRAME_LEVEL = args.frame_level

    # Defining path for feature level data
    LEVEL = 'frame_sample' if FRAME_LEVEL else 'video_sample'

    # Define path to save
    FOLDER_TO_SAVE = (PATH_ROOT / 'data' / 'raw' / f'yt8m_{YOUTUBE_DATASET_VERSION}_full' / LEVEL).__str__()

    # Check if folder exist
    if not os.path.exists(FOLDER_TO_SAVE):
        os.makedirs(FOLDER_TO_SAVE)
        
    # Change currest work directory
    cmd = f"cd {FOLDER_TO_SAVE}"
    os.chdir(FOLDER_TO_SAVE)
    
    # print saving folder
    print(f"Saving data in:     {os.getcwd()}\n")
    
    print(f"Downloading data for Youtube8M version_{YOUTUBE_DATASET_VERSION} | LEVEL_{LEVEL}\n")
    if FRAME_LEVEL is False:
        LEVEL_FEATURE = 'frame'if FRAME_LEVEL
        # Download data (executing CMD command)
        print("Dowloading training data ...")
        cmd = f"curl data.yt8m.org/download.py | partition=2/{LEVEL_FEATURE}/train mirror=us python"
        os.system(cmd)
        print("Finishing training data\n--------------------------------------------------------")

        print("Dowloading validation data ...")
        cmd = f"curl data.yt8m.org/download.py | partition=2/{LEVEL_FEATURE}/validate mirror=us python"
        os.system(cmd)
        print("Finishing validation data\n--------------------------------------------------------")
        
        print("Downloading testing data ...")
        cmd = f"curl data.yt8m.org/download.py | partition=2/{LEVEL_FEATURE}/test mirror=us python"
        os.system(cmd)
        print("Finishing testing data\n--------------------------------------------------------")

    print("Done.")