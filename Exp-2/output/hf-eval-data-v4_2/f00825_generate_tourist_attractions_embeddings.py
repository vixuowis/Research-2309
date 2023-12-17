# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import torch

# function_code --------------------

def generate_tourist_attractions_embeddings(question: str) -> torch.Tensor:
    """
    Generate the passage embedding for a tourist attraction question.

    Args:
        question (str): The question about tourist attractions.

    Returns:
        torch.Tensor: The passage embedding for given question.

    Raises:
        ValueError: If the question is empty or not a string.
    """
    if not question or not isinstance(question, str):
        raise ValueError('The question must be a non-empty string.')

    tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
    model = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')

    input_ids = tokenizer(question, return_tensors='pt')['input_ids']
    question_embedding = model(input_ids).pooler_output
    return question_embedding

# test_function_code --------------------

def test_generate_tourist_attractions_embeddings():
    print("Testing started.")

    # Test case 1: Valid question string
    print("Testing case [1/3] started.")
    valid_question = "What are the best attractions in Paris?"
    embeddings = generate_tourist_attractions_embeddings(valid_question)
    assert embeddings is not None and embeddings.shape[0] == 1, f"Test case [1/3] failed: Expected 1 embedding, got {embeddings.shape[0]} instead."

    # Test case 2: Empty question string
    print("Testing case [2/3] started.")
    empty_question = ""
    try:
        generate_tourist_attractions_embeddings(empty_question)
        assert False, "Test case [2/3] failed: ValueError not raised for empty question."
    except ValueError as e:
        assert str(e) == 'The question must be a non-empty string.', f"Test case [2/3] failed: {e}"

    # Test case 3: Non-string question input
    print("Testing case [3/3] started.")
    non_string_question = 123
    try:
        generate_tourist_attractions_embeddings(non_string_question)
        assert False, "Test case [3/3] failed: ValueError not raised for non-string question."
    except ValueError as e:
        assert str(e) == 'The question must be a non-empty string.', f"Test case [3/3] failed: {e}"

    print("Testing finished.")

# call_test_function_line --------------------

test_generate_tourist_attractions_embeddings()