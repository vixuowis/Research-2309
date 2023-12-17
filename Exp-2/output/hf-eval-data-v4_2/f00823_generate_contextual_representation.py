# requirements_file --------------------

!pip install -U transformers torch

# function_import --------------------

from transformers import BertTokenizer, AutoModel
import torch

# function_code --------------------

def generate_contextual_representation(text):
    """
    Generates contextual representations of the given Indonesian text using IndoBERT model.

    Args:
        text (str): The Indonesian text to be processed.

    Returns:
        torch.Tensor: Contextual word embeddings of the input text.

    Raises:
        ValueError: If the input text is not provided.
    """
    if not text:
        raise ValueError('Input text is not provided')

    tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
    model = AutoModel.from_pretrained('indobenchmark/indobert-base-p1')

    encoded_input = tokenizer.encode(text, return_tensors='pt')
    contextual_representation = model(encoded_input)[0]

    return contextual_representation

# test_function_code --------------------

def test_generate_contextual_representation():
    print('Testing started.')

    # Testing case: Empty text should raise ValueError
    print('Testing case [1/2] started.')
    try:
        generate_contextual_representation('')
        assert False, 'Test case [1/2] failed: ValueError was not raised for empty input.'
    except ValueError as ve:
        assert str(ve) == 'Input text is not provided', 'Test case [1/2] failed: Wrong error message.'

    # Testing case: Sample Indonesian text
    print('Testing case [2/2] started.')
    sample_text = 'Saya ingin informasi tentang wisata di Indonesia.'
    representation = generate_contextual_representation(sample_text)
    assert isinstance(representation, torch.Tensor), 'Test case [2/2] failed: The output is not a torch.Tensor.'
    print('Testing finished.')

# call_test_function_line --------------------

test_generate_contextual_representation()