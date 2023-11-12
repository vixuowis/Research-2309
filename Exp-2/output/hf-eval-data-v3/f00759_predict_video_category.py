# function_import --------------------

from transformers import XClipModel, XClipProcessor

# function_code --------------------

def predict_video_category(text: str, category_mapping: dict) -> str:
    '''
    Given a text description of a video, this function predicts the category of the video.

    Args:
        text (str): The text description of the video.
        category_mapping (dict): A dictionary mapping category indices to category names.

    Returns:
        str: The predicted category of the video.

    Raises:
        ImportError: If the necessary libraries cannot be imported.
    '''
    model = XClipModel.from_pretrained('microsoft/xclip-base-patch32')
    processor = XClipProcessor.from_pretrained('microsoft/xclip-base-patch32')

    input_text = processor(text=text, return_tensors='pt')
    output = model(**input_text)
    category_scores = output.logits.softmax(dim=-1).tolist()[0]

    predicted_category_index = category_scores.index(max(category_scores))
    predicted_category = category_mapping[predicted_category_index]
    predicted_probability = category_scores[predicted_category_index]

    return predicted_category, predicted_probability

# test_function_code --------------------

def test_predict_video_category():
    '''
    Test the function predict_video_category.
    '''
    category_mapping = {0: 'sports', 1: 'music', 2: 'news', 3: 'comedy', 4: 'education'}

    text1 = 'A football match between two teams'
    category1, probability1 = predict_video_category(text1, category_mapping)
    assert category1 in category_mapping.values(), 'Test Case 1 Failed'

    text2 = 'A music concert with a large audience'
    category2, probability2 = predict_video_category(text2, category_mapping)
    assert category2 in category_mapping.values(), 'Test Case 2 Failed'

    text3 = 'A news report on recent events'
    category3, probability3 = predict_video_category(text3, category_mapping)
    assert category3 in category_mapping.values(), 'Test Case 3 Failed'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_predict_video_category()