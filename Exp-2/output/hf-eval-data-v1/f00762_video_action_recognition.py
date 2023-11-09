from decord import VideoReader, cpu
import torch
import numpy as np
from transformers import VideoMAEFeatureExtractor, VideoMAEForVideoClassification
from huggingface_hub import hf_hub_download

def video_action_recognition(file_path: str, clip_len: int = 16, frame_sample_rate: int = 4):
    """
    Function to recognize the main action in a video clip using the pretrained 'nateraw/videomae-base-finetuned-ucf101' model.
    
    Args:
        file_path (str): Path to the video file.
        clip_len (int, optional): Length of the clip to be sampled from the video. Default is 16.
        frame_sample_rate (int, optional): Frame sample rate. Default is 4.
    
    Returns:
        str: The main action recognized in the video clip.
    """
    def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
        converted_len = int(clip_len * frame_sample_rate)
        end_idx = np.random.randint(converted_len, seg_len)
        start_idx = end_idx - converted_len
        indices = np.linspace(start_idx, end_idx, num=clip_len)
        indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        return indices

    videoreader = VideoReader(file_path, num_threads=1, ctx=cpu(0))
    indices = sample_frame_indices(clip_len=clip_len, frame_sample_rate=frame_sample_rate, seg_len=len(videoreader))
    video = videoreader.get_batch(indices).asnumpy()

    feature_extractor = VideoMAEFeatureExtractor.from_pretrained('nateraw/videomae-base-finetuned-ucf101')
    model = VideoMAEForVideoClassification.from_pretrained('nateraw/videomae-base-finetuned-ucf101')

    inputs = feature_extractor(list(video), return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    predicted_label = logits.argmax(-1).item()
    return model.config.id2label[predicted_label]