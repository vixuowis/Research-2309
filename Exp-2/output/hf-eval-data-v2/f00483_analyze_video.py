# function_import --------------------

from transformers import AutoModelForVideoClassification, AutoTokenizer

# function_code --------------------

def analyze_video(video_path):
    """
    Analyze the video to recognize the activities taking place.

    Args:
        video_path (str): The path to the video file to be analyzed.

    Returns:
        A list of tuples, where each tuple contains the recognized activity and its corresponding probability.
    """
    # Load the pre-trained model and tokenizer
    model = AutoModelForVideoClassification.from_pretrained('sayakpaul/videomae-base-finetuned-ucf101-subset')
    tokenizer = AutoTokenizer.from_pretrained('sayakpaul/videomae-base-finetuned-ucf101-subset')

    # Load the video and preprocess it
    video = load_video(video_path)
    inputs = tokenizer(video, return_tensors='pt')

    # Perform video classification
    outputs = model(**inputs)
    predicted_labels = outputs.logits.argmax(dim=1)

    # Return the recognized activities and their probabilities
    return [(label, prob) for label, prob in zip(predicted_labels, outputs.logits.softmax(dim=1))]

# test_function_code --------------------

def test_analyze_video():
    """
    Test the analyze_video function.
    """
    # Define the path to the test video
    video_path = 'test_video.mp4'

    # Call the analyze_video function
    result = analyze_video(video_path)

    # Check the result
    assert isinstance(result, list), 'The result should be a list.'
    for item in result:
        assert isinstance(item, tuple), 'Each item in the result should be a tuple.'
        assert len(item) == 2, 'Each tuple should contain two elements.'
        assert isinstance(item[0], int), 'The first element of each tuple should be an integer.'
        assert isinstance(item[1], float), 'The second element of each tuple should be a float.'

# call_test_function_code --------------------

test_analyze_video()