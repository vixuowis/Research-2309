# requirements_file --------------------

!pip install -U transformers torch

# function_import --------------------

from transformers import XClipModel, XClipTokenizer
import torch

# function_code --------------------

def classify_movie_genre(video_data, text_input):
    model = XClipModel.from_pretrained('microsoft/xclip-base-patch16-zero-shot')
    tokenizer = XClipTokenizer.from_pretrained('microsoft/xclip-base-patch16-zero-shot')

    video_inputs = torch.tensor([video_data])  # Tensor for the video data
    text_inputs = tokenizer(text_input, return_tensors='pt', padding=True)

    # Perform zero-shot video classification
    with torch.no_grad():
        outputs = model(video_inputs, **text_inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
    return probs

# test_function_code --------------------

def test_classify_movie_genre():
    print("Testing classify_movie_genre function.")

    # Assuming we have some video data and labels for testing
    test_video_data = torch.randn(3, 10, 224, 224) # Randomly generated video data tensor
    test_text_input = 'Action, Adventure, Animation, Comedy, Drama, Romance'

    # Running the classification function
    probs = classify_movie_genre(test_video_data, test_text_input)
    assert probs is not None, "The function did not return any probabilities."
    assert probs.size(1) == 6, f"Expected 6 genre probabilities, got {probs.size(1)} instead."

    print("Testing completed successfully.")

test_classify_movie_genre()