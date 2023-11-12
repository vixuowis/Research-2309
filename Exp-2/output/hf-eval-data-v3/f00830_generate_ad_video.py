# function_import --------------------

from huggingface_hub import snapshot_download
from modelscope.pipelines import pipeline
from modelscope.outputs import OutputKeys
import pathlib

# function_code --------------------

def generate_ad_video(text: str) -> str:
    '''
    Generate a video based on the provided text description.

    Args:
        text: A short text description in English.

    Returns:
        The path of the generated video.

    Raises:
        ModuleNotFoundError: If the required modules are not installed.
    '''
    model_dir = pathlib.Path('weights')
    snapshot_download('damo-vilab/modelscope-damo-text-to-video-synthesis', repo_type='model', local_dir=model_dir)
    pipe = pipeline('text-to-video-synthesis', model_dir.as_posix())
    input_text = {'text': text}
    output_video_path = pipe(input_text,)[OutputKeys.OUTPUT_VIDEO]
    return output_video_path

# test_function_code --------------------

def test_generate_ad_video():
    '''
    Test the function generate_ad_video.
    '''
    test_text1 = 'A man wearing a stylish suit while walking in the city.'
    assert isinstance(generate_ad_video(test_text1), str)
    test_text2 = 'A panda eating bamboo on a rock.'
    assert isinstance(generate_ad_video(test_text2), str)
    test_text3 = 'A woman dancing in a red dress.'
    assert isinstance(generate_ad_video(test_text3), str)
    return 'All Tests Passed'

# call_test_function_code --------------------

print(test_generate_ad_video())