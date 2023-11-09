from transformers import AutoImageProcessor, TimesformerForVideoClassification
import numpy as np
import torch


def classify_sports_video(video):
    """
    This function classifies a sports video using the TimesformerForVideoClassification model from Hugging Face Transformers.
    The model has been pre-trained on the Kinetics-600 dataset.
    
    Parameters:
    video (list): A list of numpy arrays representing the video frames.
    
    Returns:
    str: The predicted class of the sports video.
    """
    # Load the processor and model
    processor = AutoImageProcessor.from_pretrained('facebook/timesformer-hr-finetuned-k600')
    model = TimesformerForVideoClassification.from_pretrained('facebook/timesformer-hr-finetuned-k600')

    # Preprocess the video frames
    inputs = processor(images=video, return_tensors='pt')

    # Make a prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get the predicted class
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]