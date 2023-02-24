# create a function to download youtube video by ID
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


import cv2
from pytube import YouTube
import numpy as np

def get_video_frames(video_id, DOWNLOAD=False, take_first_six_min=True, take_frame_every_sec=True):
    # Download the YouTube video and get the video stream
    yt = YouTube(f'https://www.youtube.com/watch?v={video_id}')
    
    # resolution 299 x 299 for Incepetion V2
    stream = yt.streams.filter(res='360p').first() # 360p:(360*640) 240p: (240*426)
    
    # Donowload video
    if DOWNLOAD:
        stream.download('.')

    # Open the video stream and initialize a VideoCapture object
    stream_url = stream.url
    
    frames, N_FRAMES, FPS = get_stream_video(stream_url, take_first_six_min, take_frame_every_sec)
    
    return frames, N_FRAMES, FPS
    
    
    
def get_stream_video(stream_url, take_first_six_min=True, take_frame_every_sec=True):
    # Open the video stream and initialize a VideoCapture object
    cap = cv2.VideoCapture(stream_url)

    # get fps
    FPS = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Get 6 minutes in frames
    six_min_frames = 360 * FPS
    
    # get number of frames
    N_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize an empty list to store the video frames
    frames = []

    
    # Loop through the video frames and append them to the frames list
    sec = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # break when take more than 6 minutes
        if take_first_six_min:
            if len(frames) >= six_min_frames:
                break

        # only take one frame each second 
        if take_frame_every_sec:
            if sec % int(FPS) == 0:
                frames.append(frame)
            sec += 1
        else:    
            frames.append(frame)

    # Release the VideoCapture object and convert the frames list to a numpy array
    cap.release()
    frames = np.array(frames) # the channel dimension are ordered  as BGR
    # change vhannel from BGR to RGB
    frames = frames[:, :, :, ::-1]
    
    return frames, N_FRAMES, FPS


def postporcess_video(Frames, N_FRAMES, FPS):
    # get the first 6 minutes of the video
    N_SECONDS_REQ = 3600
    N_SECONDS_VIDEO = int(N_FRAMES*FPS)

    # Take the first N_SECONDS video
    if N_SECONDS_VIDEO > N_SECONDS_REQ:
        Frames_post = Frames[:N_SECONDS_REQ]
    else:
        Frames_post = Frames


    # Extract one frame each seconds, gor getting 360 frames
    N_SECONDS_VIDEO =  int(Frames_post.shape[0]/FPS)
    frames = []
    for i in range(N_SECONDS_VIDEO-1):
        frames.append(Frames_post[i*FPS])
        
    return np.array(frames)


if __name__ == '__main__':
    Frames, N_FRAMES, FPS = get_video_frames('hS5CfP8n_js')
    Frames = postporcess_video( Frames, N_FRAMES, FPS)