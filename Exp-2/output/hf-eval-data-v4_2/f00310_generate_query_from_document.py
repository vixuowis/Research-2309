# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import T5Tokenizer, T5ForConditionalGeneration

# function_code --------------------

def generate_query_from_document(document):
    tokenizer = T5Tokenizer.from_pretrained('castorini/doc2query-t5-base-msmarco')
    model = T5ForConditionalGeneration.from_pretrained('castorini/doc2query-t5-base-msmarco')
    inputs = tokenizer.encode("generate query: " + document, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, num_return_sequences=1, max_length=40)
    query = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return query


# test_function_code --------------------

def test_generate_query_from_document():
    print("Testing started.")
    test_documents = ['The quick brown fox jumps over the lazy dog.', 'To be or not to be, that is the question.']
    expected_queries = ['brown fox', 'to be or not to be']

    for i, document in enumerate(test_documents):
        print(f"Testing case [{i+1}/{len(test_documents)}] started.")
        query = generate_query_from_document(document)
        assert query == expected_queries[i], f"Test case [{i+1}/{len(test_documents)}] failed: Expected {expected_queries[i]}, got {query}"
    print("Testing finished.")


# call_test_function_line --------------------

test_generate_query_from_document()