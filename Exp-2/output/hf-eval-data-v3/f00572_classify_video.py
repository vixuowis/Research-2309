# function_import --------------------

from transformers import VideoClassificationPipeline

# function_code --------------------

def classify_video(video_path: str):
    """
    Classify the content of a video using a pre-trained model.

    Args:
        video_path (str): The path to the video file to be classified.

    Returns:
        list: The categories of the video content.

    Raises:
        AttributeError: If the model attribute 'config' is not found.
    """
    video_classifier = VideoClassificationPipeline(model='hf-tiny-model-private/tiny-random-VideoMAEForVideoClassification')
    video_categories = video_classifier(video_path)
    return video_categories

# test_function_code --------------------

def test_classify_video():
    """
    Test the classify_video function with a sample video.

    Returns:
        str: 'All Tests Passed' if all assertions pass.
    """
    video_path = 'sample_video.mp4'
    categories = classify_video(video_path)
    assert isinstance(categories, list), 'The output should be a list.'
    return 'All Tests Passed'

# call_test_function_code --------------------

if __name__ == '__main__':
    print(test_classify_video())