# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import XClipModel, XClipProcessor

# function_code --------------------

def classify_video_content(text_description, model, processor, category_mapping):
    # Process the text description using the processor
    input_text = processor(text=text_description, return_tensors="pt")
    # Get the model's output
    output = model(**input_text)
    # Calculate the softmax to get the probabilities
    category_scores = output.logits.softmax(dim=-1).tolist()[0]
    # Find the predicted category index and its probability
    predicted_category_index = category_scores.index(max(category_scores))
    predicted_category = category_mapping[predicted_category_index]
    predicted_probability = category_scores[predicted_category_index]
    # Return the predicted category and its probability
    return predicted_category, predicted_probability

# test_function_code --------------------

def test_classify_video_content():
    model = XClipModel.from_pretrained('microsoft/xclip-base-patch32')
    processor = XClipProcessor.from_pretrained('microsoft/xclip-base-patch32')
    category_mapping = {0: 'sports', 1: 'music', 2: 'news', 3: 'comedy', 4: 'education'}
    
    text_description = "Highlight clips from a basketball game"
    predicted_category, predicted_probability = classify_video_content(text_description, model, processor, category_mapping)

    assert predicted_category == 'sports', f"Test failed: Expected 'sports', got {predicted_category}"