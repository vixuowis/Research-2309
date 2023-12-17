# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import T5Tokenizer, T5ForConditionalGeneration

# function_code --------------------

def generate_search_queries(documents):
    tokenizer = T5Tokenizer.from_pretrained('castorini/doc2query-t5-base-msmarco')
    model = T5ForConditionalGeneration.from_pretrained('castorini/doc2query-t5-base-msmarco')

    queries = []
    for doc in documents:
        input_text = 'generate query: ' + doc
        inputs = tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True)
        outputs = model.generate(inputs, num_return_sequences=1, max_length=40)
        queries.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
    return queries

# test_function_code --------------------

def test_generate_search_queries():
    print('Testing generate_search_queries function...')
    sample_documents = ['This is a sample document.', 'Another example document text.']
    expected_queries = ['sample query', 'example query'] # Placeholders for expected results

    generated_queries = generate_search_queries(sample_documents)

    for i, query in enumerate(generated_queries):
        assert query == expected_queries[i], f'Test case failed for document: {sample_documents[i]}'
    print('All test cases passed.')

test_generate_search_queries()