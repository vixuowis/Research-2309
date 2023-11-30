# function_import --------------------

from transformers import AutoModelForVideoClassification

# function_code --------------------

def load_and_classify_video(model_name: str, video_path: str):
    """
    Load a pre-trained model for video classification and classify a video.

    Args:
        model_name (str): The name of the pre-trained model.
        video_path (str): The path to the video file to be classified.

    Returns:
        The classification result.

    Raises:
        FileNotFoundError: If the video file does not exist.
    """
    # Load a pre-trained model for video classification.
    model = AutoModelForVideoClassification.from_pretrained(model_name, 
                                                            return_dict=True)

    # Classify the video and print out the result.
    try:
        with open(video_path, "rb") as fp:
            video = fp.read()
        
        output = model(video)
        label = output['logits'].argmax().item() + 1 # Add 1 to make it human readable.
        print("Label of the video is '{}'.".format(label))
    except FileNotFoundError:
        raise FileNotFoundError("The video file does not exist.")

# test_function_code --------------------

def test_load_and_classify_video():
    """
    Test the load_and_classify_video function.
    """
    # Test with a known model and video file
    result = load_and_classify_video('lmazzon70/videomae-base-finetuned-kinetics-finetuned-rwf2000mp4-epochs8-batch8-kb', 'test_video.mp4')
    assert isinstance(result, str), 'The result should be a string.'

    # TODO: Add more test cases

    return 'All Tests Passed'


# call_test_function_code --------------------

test_load_and_classify_video()