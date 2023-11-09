# function_import --------------------

from transformers import XClipModel, XClipProcessor

# function_code --------------------

def predict_video_category(text):
    """
    Given the text description of a video, this function predicts the category of the video.
    
    Args:
        text (str): The text description of the video.
    
    Returns:
        tuple: A tuple containing the predicted category and its probability.
    
    Raises:
        ValueError: If the input text is not a string.
    """
    if not isinstance(text, str):
        raise ValueError('Input text must be a string.')
    
    model = XClipModel.from_pretrained('microsoft/xclip-base-patch32')
    processor = XClipProcessor.from_pretrained('microsoft/xclip-base-patch32')
    
    input_text = processor(text=text, return_tensors='pt')
    output = model(**input_text)
    category_scores = output.logits.softmax(dim=-1).tolist()[0]
    
    category_mapping = {0: 'sports', 1: 'music', 2: 'news', 3: 'comedy', 4: 'education'}
    
    predicted_category_index = category_scores.index(max(category_scores))
    predicted_category = category_mapping[predicted_category_index]
    predicted_probability = category_scores[predicted_category_index]
    
    return predicted_category, predicted_probability

# test_function_code --------------------

def test_predict_video_category():
    """
    Test the function predict_video_category.
    
    Raises:
        AssertionError: If the function does not return the expected results.
    """
    text = 'A video about a football match.'
    predicted_category, predicted_probability = predict_video_category(text)
    
    assert isinstance(predicted_category, str), 'The predicted category must be a string.'
    assert isinstance(predicted_probability, float), 'The predicted probability must be a float.'
    assert 0 <= predicted_probability <= 1, 'The predicted probability must be between 0 and 1.'

# call_test_function_code --------------------

test_predict_video_category()