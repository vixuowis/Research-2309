# requirements_file --------------------

!pip install -U modelscope==1.4.2 open_clip_torch pytorch-lightning

# function_import --------------------

from huggingface_hub import snapshot_download
from modelscope.pipelines import pipeline
from modelscope.outputs import OutputKeys
import pathlib

# function_code --------------------

def generate_ad_video(description):
    """
    Generate a personalized ad video based on a text description.

    Args:
        description (str): Text description for the ad video.

    Returns:
        str: Path to the rendered ad video.
    """
    model_dir = pathlib.Path('weights')
    snapshot_download('damo-vilab/modelscope-damo-text-to-video-synthesis', repo_type='model', local_dir=model_dir)
    pipe = pipeline('text-to-video-synthesis', model_dir.as_posix())
    input_text = {'text': description}
    output_video_path = pipe(input_text,)[OutputKeys.OUTPUT_VIDEO]
    return output_video_path

# test_function_code --------------------

def test_generate_ad_video():
    print("Testing generate_ad_video function.")

    # Test case 1: Short description
    description_1 = 'A dog playing with a ball in the park.'
    output_1 = generate_ad_video(description_1)
    assert isinstance(output_1, str), f"Failed to generate video for description: {description_1}"

    # Test case 2: Descriptive text
    description_2 = 'A chef preparing a gourmet meal in a professional kitchen.'
    output_2 = generate_ad_video(description_2)
    assert isinstance(output_2, str), f"Failed to generate video for description: {description_2}"

    # Test case 3: Fashion related description
    description_3 = 'A model showcasing the latest summer collection on the runway.'
    output_3 = generate_ad_video(description_3)
    assert isinstance(output_3, str), f"Failed to generate video for description: {description_3}"
    print("Testing completed successfully.")

# Run the test for generate_ad_video function
test_generate_ad_video()