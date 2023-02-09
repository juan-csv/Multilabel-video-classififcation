""" Downloading data from youtube8M dataset
https://research.google.com/youtube8m/download.html
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
    # Create arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--')
    
    # Arguments
    YOUTUBE_DATASET_VERSION = 2 # 1, 2 or 3
    FRAME_LEVEL = False

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
    
    if FRAME_LEVEL is False:
        print(f"Downloading data for Youtube8M version_{YOUTUBE_DATASET_VERSION} | LEVEL_{LEVEL}\n")

        # Download data (executing CMD command)
        print(f"Dowloading training data ...")
        cmd = "curl data.yt8m.org/download.py | partition=2/video/train mirror=us python"
        os.system(cmd)
        print("Finishing training data\n--------------------------------------------------------")

        print(f"Dowloading validation data ...")
        cmd = "curl data.yt8m.org/download.py | partition=2/video/validate mirror=us python"
        os.system(cmd)
        print("Finishing validation data\n--------------------------------------------------------")
        
        print("Downloading testing data ...")
        cmd = "curl data.yt8m.org/download.py | partition=2/video/test mirror=us python"
        os.system(cmd)
        print("Finishing testing data\n--------------------------------------------------------")

        print("Done.")
