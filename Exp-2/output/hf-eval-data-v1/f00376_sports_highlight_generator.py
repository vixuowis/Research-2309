from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import numpy as np
import torch


def sports_highlight_generator(video):
    '''
    This function takes a video as input and identifies the category of the sports activity happening in the video.
    The video should be a list of numpy arrays representing video frames.
    The function uses the 'MCG-NJU/videomae-small-finetuned-kinetics' model from Hugging Face Transformers, which is trained on the Kinetics-400 dataset.
    '''
    # Load the model and processor
    processor = VideoMAEImageProcessor.from_pretrained('MCG-NJU/videomae-small-finetuned-kinetics')
    model = VideoMAEForVideoClassification.from_pretrained('MCG-NJU/videomae-small-finetuned-kinetics')

    # Process the video input
    inputs = processor(video, return_tensors='pt')

    # Pass the processed input to the model and obtain the output logits
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Find the index of the maximum class logits and use the model's configuration to obtain the predicted class label
    predicted_class_idx = logits.argmax(-1).item()
    predicted_class = model.config.id2label[predicted_class_idx]

    return predicted_class