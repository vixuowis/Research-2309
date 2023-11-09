# function_import --------------------

from huggingface_hub import snapshot_download
from modelscope.pipelines import pipeline
from modelscope.outputs import OutputKeys
import pathlib

# function_code --------------------

def generate_ad_video(input_text):
    """
    This function generates a video based on the input text description using the 'modelscope-damo-text-to-video-synthesis' model.

    Args:
        input_text (dict): A dictionary containing a key 'text' with a short text description in English as the value.

    Returns:
        str: The path to the output video.
    """
    model_dir = pathlib.Path('weights')
    snapshot_download('damo-vilab/modelscope-damo-text-to-video-synthesis', repo_type='model', local_dir=model_dir)
    pipe = pipeline('text-to-video-synthesis', model_dir.as_posix())
    output_video_path = pipe(input_text,)[OutputKeys.OUTPUT_VIDEO]
    return output_video_path

# test_function_code --------------------

def test_generate_ad_video():
    """
    This function tests the 'generate_ad_video' function by providing a test text and checking if the output is a valid file path.
    """
    test_text = {'text': 'A panda eating bamboo on a rock.'}
    output_video_path = generate_ad_video(test_text)
    assert pathlib.Path(output_video_path).is_file(), 'The output should be a valid file path.'

# call_test_function_code --------------------

test_generate_ad_video()