# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import T5Tokenizer, T5ForConditionalGeneration

# function_code --------------------

def generate_possible_user_queries(document_text):
    # Initializes the tokenizer and model from the pretrained castorini/doc2query-t5-base-msmarco
    tokenizer = T5Tokenizer.from_pretrained('castorini/doc2query-t5-base-msmarco')
    model = T5ForConditionalGeneration.from_pretrained('castorini/doc2query-t5-base-msmarco')

    # Tokenize the document text and generate queries
    input_ids = tokenizer.encode(document_text, return_tensors='pt', add_special_tokens=True)
    generated_ids = model.generate(input_ids)

    # Decode the generated tokens to get the queries
    queries = [tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]

    return queries

# test_function_code --------------------

def test_generate_possible_user_queries():
    print("Testing started.")
    sample_document = "What is Natural Language Processing?"

    # Test case: Verifying generated queries are strings
    print("Testing case [1/1] started.")
    generated_queries = generate_possible_user_queries(sample_document)
    assert all(isinstance(query, str) for query in generated_queries), "Test case failed: Generated queries are not all strings."
    print("Testing case [1/1] finished.")
    print("Testing finished.")

test_generate_possible_user_queries()