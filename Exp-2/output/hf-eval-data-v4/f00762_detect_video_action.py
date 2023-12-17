# requirements_file --------------------

!pip install -U transformers, decord, huggingface_hub

# function_import --------------------

from decord import VideoReader, cpu
import torch
import numpy as np
from transformers import VideoMAEFeatureExtractor, VideoMAEForVideoClassification
from huggingface_hub import hf_hub_download

# function_code --------------------

def detect_video_action(video_path: str) -> str:
    """
    Detect the main action in a video file using a pretrained VideoMAE model.

    Parameters:
        video_path (str): The file path to the video.

    Returns:
        str: The predicted action label.
    """
    # Function parameters, such as clip length and frame sampling, could be
    # adjusted to optimize performance for different video types.
    clip_len = 16
    frame_sample_rate = 4

    # Sample frame indices
    def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
        converted_len = int(clip_len * frame_sample_rate)
        end_idx = np.random.randint(converted_len, seg_len)
        start_idx = end_idx - converted_len
        indices = np.linspace(start_idx, end_idx, num=clip_len)
        indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        return indices

    video_reader = VideoReader(video_path, num_threads=1, ctx=cpu(0))
    indices = sample_frame_indices(clip_len, frame_sample_rate, len(video_reader))
    video = video_reader.get_batch(indices).asnumpy()

    # Load the pretrained model
    feature_extractor = VideoMAEFeatureExtractor.from_pretrained('nateraw/videomae-base-finetuned-ucf101')
    model = VideoMAEForVideoClassification.from_pretrained('nateraw/videomae-base-finetuned-ucf101')

    # Prepare inputs and perform inference
    inputs = feature_extractor(list(video), return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get the predicted label
    predicted_label = logits.argmax(-1).item()
    # Map label ID to label name
    action_label = model.config.id2label[predicted_label]
    return action_label


# test_function_code --------------------

def test_detect_video_action():
    print("Testing detect_video_action function.")

    test_video_path = hf_hub_download('archery.mp4')

    # Testing the action detection
    print("Testing action detection.")
    predicted_action = detect_video_action(test_video_path)
    assert predicted_action.isalpha(), "The predicted action label should be a string."

    # More test cases could include verifying the output against known video labels
    # and testing the function with videos of varying lengths and qualities.
    print("Testing completed successfully.")

# Run the test function
test_detect_video_action()
