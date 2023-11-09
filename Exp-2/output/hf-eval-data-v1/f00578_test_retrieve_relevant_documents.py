def test_retrieve_relevant_documents():
    '''
    This function tests the 'retrieve_relevant_documents' function with a sample query and documents.
    '''
    query = 'How many people live in Berlin?'
    documents = ['Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.', 'New York City is famous for the Metropolitan Museum of Art.']
    
    expected_output = ['Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.', 'New York City is famous for the Metropolitan Museum of Art.']
    
    assert retrieve_relevant_documents(query, documents) == expected_output, 'Test failed!'
    
    print('Test passed!')

test_retrieve_relevant_documents()