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
import streamlit as st

# Local imports
from src.model.inference_video_level import ModelTag, get_path_model
from src.data.get_youtube_video import get_video_frames, postporcess_video



# load config
PATH_CONFIG = PATH_ROOT / 'config_files' / 'config.yaml'
with open( PATH_CONFIG ) as f:
    config = yaml.safe_load( f )


# Instance Model
RUN_ID = '02_14_10_15'
LEVEL_FEATURE = 'video'
FEATURES = ['rgb'] # ['audio', 'rgb']
NAME_EXPERIMENT = f"{RUN_ID}_baseline_{LEVEL_FEATURE}-level_{'_'.join(FEATURES)}"
MODEL = 'LinearModel'
FOLDERS_SAVE_MODEL =        PATH_ROOT / config['Model']['folder']
PATH_MODEL = get_path_model(FOLDERS_SAVE_MODEL, LEVEL_FEATURE, MODEL, RUN_ID)
# Load model
model_video = ModelTag(PATH_MODEL)


# Set page title
st.set_page_config(page_title="Classification content video")

# Define function to get video ID from YouTube URL
def get_video_id(url):
    # Check if URL is a valid YouTube video URL
    if "youtube.com/watch?v=" not in url:
        st.error("Invalid YouTube video URL. Please enter a valid URL.")
        return
    # Extract video ID from URL
    video_id = url.split("youtube.com/watch?v=")[-1]
    return video_id

# Define Streamlit app

# Set app title
st.title("YouTube Video Embedding App")

# Get user input for YouTube video URL
video_url = st.text_input("Enter a YouTube video URL: ")

# Check if user has entered a video URL
if video_url:
    # Get video ID from URL
    video_id = get_video_id(video_url)
    if video_id:
        # Create embedded video URL
        embed_url = f"https://www.youtube.com/embed/{video_id}"
        # Display embedded video
        st.video(embed_url)

        if st.button("Get video categories"):
            # add animation for loading
            with st.spinner("Downloading video ..."):
                # Get video categories using YouTube Data API
                Frames, N_FRAMES, FPS = get_video_frames(video_id)
                Frames = postporcess_video( Frames, N_FRAMES, FPS)
            
            with st.spinner("Calculating categories ..."):
                label_pred, score_pred = model_video(Frames)

            # show the categories
            st.info(label_pred[0])

                

