import decord
from decord import VideoReader, cpu, gpu
import numpy as np
import torch
from transformers import AutoProcessor, AutoModel

decord.bridge.set_bridge('torch')


# returns split_frames indices that evenly divide total_frames
def split_video_indices(split_frames, total_frames):
    return np.linspace(0, total_frames - 1, num=split_frames).astype(np.int64)


video_path = "dataset/vatex_test_videos/_0ZBlXUcaOk_000013_000023.mp4"
with open(video_path, "rb") as video_file:
    vr = VideoReader(video_file, ctx=cpu(0))

indices = split_video_indices(8, len(vr))
video = vr.get_batch(indices)

processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
model = AutoModel.from_pretrained("microsoft/xclip-base-patch32")

inputs = processor(videos=list(video), return_tensors="pt")
video_features = model.get_video_features(**inputs)
print(video_features.shape)
print(video_features)

# inputs = processor(
#     text=["playing sports", "eating spaghetti", "go shopping", "doing gymnastics"],
#     videos=list(video),
#     return_tensors="pt",
#     padding=True,
# )

# with torch.no_grad():
#     outputs = model(**inputs)

# logits_per_video = outputs.logits_per_video  # video-text similarity score
# probs = logits_per_video.softmax(dim=1)  # label probabilities
# print(probs)