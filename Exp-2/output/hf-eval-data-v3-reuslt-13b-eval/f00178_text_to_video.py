# function_import --------------------

from transformers import pipeline

# function_code --------------------

def text_to_video(scene_description: str) -> None:
    """
    Convert a scene description from a script into a video.

    Args:
        scene_description (str): The scene description from the script.

    Returns:
        None

    Raises:
        Exception: If the model fails to generate a video.
    """
    
    # function_header --------------------

    print("::group::Text-to-Video")
    print("Converting scene description into text-to-video...\n")

    try:
        text2vid = pipeline("text-to-image", model="gagan3012/VQGAN-CLIP.1-512x48") # load the model
        text2vid(scene_description)
        print("\nVideo created successfully!")
    except Exception as e:
        print("An unexpected error occured while generating video...\n" + str(e))
    
    print("::endgroup::")

# function_import --------------------

# test_function_code --------------------

def test_text_to_video():
    """
    Test the text_to_video function.
    """
    scene_description = 'Scene description from the script...'
    try:
        text_to_video(scene_description)
        print('Test passed')
    except Exception as e:
        print(f'Test failed: {e}')


# call_test_function_code --------------------

test_text_to_video()