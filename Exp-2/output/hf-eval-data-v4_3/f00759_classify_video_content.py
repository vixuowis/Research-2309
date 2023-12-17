# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import XClipModel, XClipProcessor

# function_code --------------------

def classify_video_content(text_description, model_name='microsoft/xclip-base-patch32'):
    """
    Classify the content of a video based on a text description.

    Args:
        text_description (str): A text description of the video content.
        model_name (str): The model name for the pre-trained XClipModel. Default to 'microsoft/xclip-base-patch32'.

    Returns:
        tuple: A tuple containing the predicted category as a string and its probability as a float.
    """
    processor = XClipProcessor.from_pretrained(model_name)
    model = XClipModel.from_pretrained(model_name)
    input_text = processor(text=text_description, return_tensors="pt")
    output = model(**input_text)
    category_scores = output.logits.softmax(dim=-1).tolist()[0]

    # Mapping category indices to generic names
    category_mapping = {0: 'sports', 1: 'music', 2: 'news', 3: 'comedy', 4: 'education'}

    # Extract top category
    predicted_category_index = category_scores.index(max(category_scores))
    predicted_category = category_mapping[predicted_category_index]
    predicted_probability = category_scores[predicted_category_index]

    return (predicted_category, predicted_probability)

# test_function_code --------------------

def test_classify_video_content():
    print("Testing started.")

    # Test case 1: Sports video description
    print("Testing case [1/3] started.")
    sports_text = "A basketball game with amazing dunks."
    category, probability = classify_video_content(sports_text)
    assert category == 'sports', f"Test case [1/3] failed: Expected 'sports', got {category}"

    # Test case 2: Music video description
    print("Testing case [2/3] started.")
    music_text = "A new music video from a popular band."
    category, probability = classify_video_content(music_text)
    assert category == 'music', f"Test case [2/3] failed: Expected 'music', got {category}"

    # Test case 3: Educational video description
    print("Testing case [3/3] started.")
    educational_text = "A tutorial on how to code in Python."
    category, probability = classify_video_content(educational_text)
    assert category == 'education', f"Test case [3/3] failed: Expected 'education', got {category}"

    print("Testing finished.")

# call_test_function_line --------------------

test_classify_video_content()