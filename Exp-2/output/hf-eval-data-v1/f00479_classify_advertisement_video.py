from transformers import AutoImageProcessor, TimesformerForVideoClassification
import numpy as np
import torch

def classify_advertisement_video(video):
    '''
    Classify the category of an advertisement video using a pre-trained model from Hugging Face Transformers.
    
    Parameters:
    video (list): A list of 3D numpy arrays representing the video frames.
    
    Returns:
    str: The predicted class of the advertisement video.
    '''
    # Load the pre-trained model and processor
    processor = AutoImageProcessor.from_pretrained('facebook/timesformer-base-finetuned-k600')
    model = TimesformerForVideoClassification.from_pretrained('facebook/timesformer-base-finetuned-k600')
    
    # Process the input video
    inputs = processor(images=video, return_tensors='pt')
    
    # Pass the processed inputs through the model and obtain the logits
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Find the predicted class index with the highest logits value
    predicted_class_idx = logits.argmax(-1).item()
    
    # Return the predicted class
    return model.config.id2label[predicted_class_idx]