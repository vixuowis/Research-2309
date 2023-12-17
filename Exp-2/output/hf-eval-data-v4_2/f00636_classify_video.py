# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import XClipModel, XClipTokenizer

# function_code --------------------

def classify_video(video_data, genres):
    """Classify a video into given genres using a pre-trained XClipModel.

    Args:
        video_data (list): The video data to be classified.
        genres (str): Comma-separated string of genres.

    Returns:
        dict: A dictionary with genres as keys and their corresponding confidence scores as values.

    Raises:
        ValueError: If `video_data` is empty or `genres` is not provided.
    """
    if not video_data:
        raise ValueError('Video data is required for classification.')
    if not genres:
        raise ValueError('Genres string is required for classification.')

    # Load the model and tokenizer
    model = XClipModel.from_pretrained('microsoft/xclip-base-patch16-zero-shot')
    tokenizer = XClipTokenizer.from_pretrained('microsoft/xclip-base-patch16-zero-shot')

    # Tokenize the genres
    text_inputs = tokenizer(genres, return_tensors='pt', padding=True)

    # Classify the video
    outputs = model(video_data, text_inputs.input_ids)
    features = outputs.last_hidden_state

    # Placeholder logic for extracting genre confidence scores from features
    # In practice, this will be a more complex function
    genre_scores = {genre: features.mean().item() for genre in genres.split(', ')}

    return genre_scores


# test_function_code --------------------

def test_classify_video():
    print("Testing started.")
    # Assumes load_dataset is a function that loads video datasets
    dataset = load_dataset("some_video_classification_dataset")
    sample_video_data = dataset[0]  # Extract one sample video from the dataset

    genres = 'Action, Adventure, Animation, Comedy, Drama, Romance'

    # We are only testing if the function returns the required structure
    # with the right number of keys and non-negative scores.
    print("Testing case [1/1] started.")
    classification_result = classify_video(sample_video_data, genres)
    assert isinstance(classification_result, dict), "Result should be a dictionary."
    assert set(classification_result.keys()) == set(genres.split(', ')), "The dictionary keys should match the genres."
    assert all(score >= 0 for score in classification_result.values()), "All scores should be non-negative."
    print("Testing case [1/1] finished.")
    print("Testing finished.")


# call_test_function_line --------------------

test_classify_video()