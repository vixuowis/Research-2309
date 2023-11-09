# function_import --------------------

from transformers import T5Tokenizer, T5ForConditionalGeneration

# function_code --------------------

def generate_query(document):
    """
    Generate a query from a given document using a pre-trained T5 model.

    Args:
        document (str): The document from which to generate the query.

    Returns:
        str: The generated query.
    """
    tokenizer = T5Tokenizer.from_pretrained('castorini/doc2query-t5-base-msmarco')
    model = T5ForConditionalGeneration.from_pretrained('castorini/doc2query-t5-base-msmarco')
    inputs = tokenizer.encode("generate query: " + document, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, num_return_sequences=1, max_length=40)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# test_function_code --------------------

def test_generate_query():
    """
    Test the generate_query function.

    Raises:
        AssertionError: If the function does not return a string.
    """
    document = 'This is a test document.'
    query = generate_query(document)
    assert isinstance(query, str), 'The function should return a string.'

# call_test_function_code --------------------

test_generate_query()