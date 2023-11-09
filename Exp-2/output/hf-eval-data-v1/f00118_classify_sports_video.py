from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import numpy as np
import torch

def classify_sports_video(video):
    '''
    This function classifies the input video into one of the sports categories using a pre-trained model.
    The model has been fine-tuned on the Kinetics-400 dataset.
    
    Parameters:
    video (list): A list of video frames. Each frame is a 3D numpy array with shape (3, 224, 224).
    
    Returns:
    str: The predicted sports category.
    '''
    processor = VideoMAEImageProcessor.from_pretrained('MCG-NJU/videomae-large-finetuned-kinetics')
    model = VideoMAEForVideoClassification.from_pretrained('MCG-NJU/videomae-large-finetuned-kinetics')
    inputs = processor(video, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]