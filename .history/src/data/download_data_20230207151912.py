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



PATH_TO_SAVE = PATH_ROOT