from transformers import AutoImageProcessor, TimesformerForVideoClassification
import numpy as np
import torch


def video_classification(video_path):
    """
    This function classifies the content of a video lecture using a pre-trained Timesformer model.
    
    Parameters:
    video_path (str): The path to the video file.
    
    Returns:
    str: The predicted class of the video content.
    """
    # replace with a function that will extract video frames as a list of images
    video = get_video_frames(video_path)  
    
    # Initialize the image processor and the model
    processor = AutoImageProcessor.from_pretrained('fcakyon/timesformer-hr-finetuned-k400')
    model = TimesformerForVideoClassification.from_pretrained('fcakyon/timesformer-hr-finetuned-k400')
    
    # Process the video frames
    inputs = processor(images=video, return_tensors='pt')
    
    # Predict the class of the video content
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Get the index of the predicted class
    predicted_class_idx = logits.argmax(-1).item()
    
    # Return the predicted class
    return model.config.id2label[predicted_class_idx]