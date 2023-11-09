from transformers import VideoClassificationPipeline


def classify_video(video_path):
    """
    This function classifies the content of a video using a tiny random VideoMAE model for video classification.
    The model is provided by Hugging Face Transformers.
    Note: The accuracy of this model may not be as high as more advanced models.
    
    Args:
    video_path (str): The path to the video file to be classified.
    
    Returns:
    video_categories (list): The list of predicted categories for the video.
    """
    # Create an instance of VideoClassificationPipeline using the specified model
    video_classifier = VideoClassificationPipeline(model='hf-tiny-model-private/tiny-random-VideoMAEForVideoClassification')
    
    # Use the classifier to predict the categories of the video
    video_categories = video_classifier(video_path)
    
    return video_categories