import os

import cv2
from datasets import load_dataset
import numpy as np
from transformers import AutoTokenizer

FRAMES_PER_VIDEO = 16

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def preprocess(example):
    video_id = example["videoID"]
    captions = example["enCap"]
    
    videos_path = "dataset/videos"
    video_path = os.path.join(videos_path, "%s.mp4" % video_id)
    if not os.path.isfile(video_path):
        video_path = os.path.join(videos_path, "%s.webm" % video_id)
    
    # count number of frames
    video = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, _ = video.read()
        if not ret:
            break
        frame_count += 1
    video.release()
        
    # fixed frame sampling
    indices = np.linspace(0, frame_count, num=FRAMES_PER_VIDEO, endpoint=False).astype(np.int64)
    # random frame sampling
    #indices = np.sort(np.random.uniform(low=0, high=frame_count, size=self.num_frames).astype(np.int64))
    
    # get frames
    video = cv2.VideoCapture(video_path)
    frames = []
    frame_count, frame_idx = 0, 0
    while frame_idx < len(indices):
        if frame_count == indices[frame_idx]:
            _, frame = video.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            frame_idx += 1
        else:
            video.grab()
        frame_count += 1
    video.release()
        
    # longest caption
    max_len = -np.inf
    caption = None
    for cap in captions:
        length = len(cap.split(" "))
        if length > max_len:
            max_len = length
            caption = cap
    # random caption
    #caption = captions[random.randint(0, 9)]

    labels = tokenizer(caption, padding="max_length").input_ids
    return {"pixel_values": frames, "labels": labels}

data_files = {"train": "dataset/vatex_train_captions.json", "validation": "dataset/vatex_val_captions.json"}
dataset = load_dataset("json", data_files=data_files)
dataset = dataset.map(function=preprocess, remove_columns=["enCap", "chCap"])
dataset.save_to_disk("dataset/raw_frames_16")