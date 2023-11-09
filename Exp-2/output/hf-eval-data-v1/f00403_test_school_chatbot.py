def test_school_chatbot():
    # Define a set of queries for testing
    test_queries = [
        'What is the admission process for the new academic year?',
        'Who are the teachers for grade 10?',
        'What extracurricular activities are available?',
        'When do classes start?'
    ]

    # Generate responses for each query
    for query in test_queries:
        response = school_chatbot(query)
        print(f'Query: {query}')
        print(f'Response: {response}')
        assert response is not None and isinstance(response, str)

test_school_chatbot()