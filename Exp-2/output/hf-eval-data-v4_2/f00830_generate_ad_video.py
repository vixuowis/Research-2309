# requirements_file --------------------

!pip install -U huggingface_hub modelscope==1.4.2 open_clip_torch pytorch-lightning

# function_import --------------------

from huggingface_hub import snapshot_download
from modelscope.pipelines import pipeline
from modelscope.outputs import OutputKeys
import pathlib

# function_code --------------------

def generate_ad_video(description_text):
    """
    Generates an advertising video from a text description using the Hugging Face multimodal synthesis model.

    Args:
        description_text (str): A text description of the content for the advertising video.

    Returns:
        str: Path to the generated video file.

    Raises:
        ValueError: If the input text description is empty or None.
        RuntimeError: If the video generation pipeline encounters an error.
    """
    # Validate input text description
    if not description_text:
        raise ValueError('Input text description is required.')

    # Define model directory path
    model_dir = pathlib.Path('weights')
    # Download the pretrained model
    snapshot_download('damo-vilab/modelscope-damo-text-to-video-synthesis', repo_type='model', local_dir=model_dir)

    # Create a video generation pipeline
    pipe = pipeline('text-to-video-synthesis', model_dir.as_posix())

    # Execute the pipeline to generate a video
    gen_video = pipe({'text': description_text})[OutputKeys.OUTPUT_VIDEO]

    # Check if the video was generated successfully
    if not gen_video:
        raise RuntimeError('Video generation failed.')

    return gen_video

# test_function_code --------------------

def test_generate_ad_video():
    print("Testing started.")
    # Test case 1: Valid text description
    print("Testing case [1/3] started.")
    gen_video = generate_ad_video('A man wearing a stylish suit while walking in the city.')
    assert pathlib.Path(gen_video).exists(), f"Test case [1/3] failed: Expected video file, got {gen_video}"

    # Test case 2: Empty text description
    print("Testing case [2/3] started.")
    try:
        generate_ad_video('')
        assert False, "Test case [2/3] failed: ValueError expected for empty description."
    except ValueError:
        pass

    # Test case 3: None as text description
    print("Testing case [3/3] started.")
    try:
        generate_ad_video(None)
        assert False, "Test case [3/3] failed: ValueError expected for None description."
    except ValueError:
        pass
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_ad_video()