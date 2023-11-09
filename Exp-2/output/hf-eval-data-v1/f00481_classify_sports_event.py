from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import numpy as np
import torch


def classify_sports_event(video):
    '''
    Function to classify sports event in a video using a pre-trained model from Hugging Face Transformers.
    The model used is 'MCG-NJU/videomae-base-short-finetuned-kinetics' which is trained on the Kinetics-400 dataset.
    
    Parameters:
    video (list): The video data to be classified.
    
    Returns:
    str: The predicted class of the sports event.
    '''
    
    # Load the pre-trained model and the processor
    processor = VideoMAEImageProcessor.from_pretrained('MCG-NJU/videomae-base-short-finetuned-kinetics')
    model = VideoMAEForVideoClassification.from_pretrained('MCG-NJU/videomae-base-short-finetuned-kinetics')
    
    # Process the video data
    inputs = processor(video, return_tensors='pt')
    
    # Make the prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Get the predicted class
    predicted_class_idx = logits.argmax(-1).item()
    
    return model.config.id2label[predicted_class_idx]