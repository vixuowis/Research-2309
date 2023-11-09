from transformers import XClipModel, XClipProcessor


def predict_video_category(text):
    """
    Given a text description of a video, this function uses the XClipModel to estimate the content of the video and predict its category.

    Args:
        text (str): Text description of a video.

    Returns:
        str: Predicted category of the video.
        float: Probability of the predicted category.
    """
    model = XClipModel.from_pretrained('microsoft/xclip-base-patch32')
    processor = XClipProcessor.from_pretrained('microsoft/xclip-base-patch32')

    input_text = processor(text=text, return_tensors="pt")
    output = model(**input_text)
    category_scores = output.logits.softmax(dim=-1).tolist()[0]

    category_mapping = {0: 'sports', 1: 'music', 2: 'news', 3: 'comedy', 4: 'education'}

    predicted_category_index = category_scores.index(max(category_scores))
    predicted_category = category_mapping[predicted_category_index]
    predicted_probability = category_scores[predicted_category_index]

    return predicted_category, predicted_probability