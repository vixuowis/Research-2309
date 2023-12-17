# requirements_file --------------------

!pip install -U torch, transformers

# function_import --------------------

from transformers import AutoTokenizer, AutoModel

# function_code --------------------

def extract_code_features(code_snippet):
    # Initialize the tokenizer and model from Hugging Face Transformers
    tokenizer = AutoTokenizer.from_pretrained('microsoft/unixcoder-base')
    model = AutoModel.from_pretrained('microsoft/unixcoder-base')

    # Tokenize the code snippet
    inputs = tokenizer(code_snippet, return_tensors='pt')

    # Generate the code embedding
    with torch.no_grad():
        outputs = model(**inputs)

    # Retrieve the embeddings from the model's output
    embeddings = outputs.last_hidden_state
    return embeddings

# test_function_code --------------------

def test_extract_code_features():
    print("Testing extract_code_features function.")

    # Sample code snippet to test
    sample_code = 'def add(x, y):\n    return x + y'
    embeddings = extract_code_features(sample_code)

    # Test case: Check if embeddings are not None
    assert embeddings is not None, 'Embeddings should not be None'

    # Test case: Check the shape of embeddings
    assert embeddings.shape[1] == 768, 'Embedding size should be 768'

    # If no assert is raised, print success message
    print("All tests passed!")

# Run the test function
test_extract_code_features()