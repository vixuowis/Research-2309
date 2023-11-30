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
    # load model and prepare it for evaluation 
    model = AutoModelForVideoClassification.from_pretrained(model_name)
    model.eval()
    
    # check whether the file exists
    from pathlib import Path
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError("{} is not a valid video file".format(video_path))  

    # prepare input tensor for classification
    from transformers import VideoUtils
    from PIL import Image
    from io import BytesIO
    
    # load raw video data and extract the first frame
    video = VideoUtils.load_video(str(video_path))[0]["video"]
    video_frame = Image.open(BytesIO(video)).convert("RGB")
    video_feature = VideoUtils.resize_and_center_crop(video_frame, size=112)
    
    # load the model's input tokenizer
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    inputs = feature_extractor(images=[video_feature], return_tensors="pt")

    # classify video and extract top 5 predicted labels
    outputs = model(**inputs)  
    logits = outputs.logits[0]
    _, indices = torch.topk(outputs["logits"], k=5, dim=-1)
    
    return indices, logits


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