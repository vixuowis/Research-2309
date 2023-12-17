# requirements_file --------------------

!pip install -U transformers torch

# function_import --------------------

from transformers import BertTokenizer, AutoModel
import torch

# function_code --------------------

def generate_contextual_representation(text):
    tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
    model = AutoModel.from_pretrained('indobenchmark/indobert-base-p1')

    encoded_input = tokenizer.encode(text, return_tensors='pt')
    contextual_representation = model(encoded_input)[0]
    return contextual_representation

# test_function_code --------------------

def test_generate_contextual_representation():
    print("Testing started.")

    # Test case 1: Check if the output is a torch tensor
    print("Testing case [1/3] started.")
    sample_text = 'Ini adalah contoh teks dalam bahasa Indonesia.'
    representation = generate_contextual_representation(sample_text)
    assert isinstance(representation, torch.Tensor), f"Test case [1/3] failed: Expected torch.Tensor, got {type(representation)}"

    # Test case 2: Check the shape of the representation
    print("Testing case [2/3] started.")
    expected_shape = (1, 768) # Assuming that 'encoded_input' would be a single sentence with a hidden size of 768
    assert representation.shape[1] == expected_shape[1], f"Test case [2/3] failed: Expected shape {expected_shape}, got {representation.shape}"

    # Test case 3: Check if the function handles empty string
    print("Testing case [3/3] started.")
    empty_text = ''
    representation = generate_contextual_representation(empty_text)
    expected_shape = (1, 768)
    assert representation.shape == expected_shape, f"Test case [3/3] failed: Expected shape {expected_shape}, got {representation.shape}"
    print("Testing finished.")

# Run test function
test_generate_contextual_representation()