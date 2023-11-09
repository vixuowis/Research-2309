def test_get_most_suitable_response():
    query = 'How many people live in London?'
    docs = ['Around 9 Million people live in London', 'London is known for its financial district']
    assert get_most_suitable_response(query, docs) == 'Around 9 Million people live in London'

test_get_most_suitable_response()