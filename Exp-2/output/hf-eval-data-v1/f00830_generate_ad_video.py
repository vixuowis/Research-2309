from huggingface_hub import snapshot_download
from modelscope.pipelines import pipeline
from modelscope.outputs import OutputKeys
import pathlib

def generate_ad_video(text):
    """
    Generate a video for an ad campaign based on the provided text description.

    Args:
        text (str): A short text description in English of the desired video.

    Returns:
        str: The path to the generated video.
    """
    model_dir = pathlib.Path('weights')
    snapshot_download('damo-vilab/modelscope-damo-text-to-video-synthesis', repo_type='model', local_dir=model_dir)
    pipe = pipeline('text-to-video-synthesis', model_dir.as_posix())
    input_text = {'text': text}
    output_video_path = pipe(input_text,)[OutputKeys.OUTPUT_VIDEO]
    return output_video_path