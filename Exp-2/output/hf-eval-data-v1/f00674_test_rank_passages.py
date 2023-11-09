def test_rank_passages():
    '''
    This function tests the rank_passages function.
    It uses a sample query and passages, and checks if the function returns a list.
    '''
    query = 'How many people live in Berlin?'
    passages = ['Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.', 'New York City is famous for the Metropolitan Museum of Art.']
    ranked_passages = rank_passages(query, passages)
    assert isinstance(ranked_passages, list), 'The function should return a list.'
    assert len(ranked_passages) == len(passages), 'The function should return a list of the same length as the input passages.'

test_rank_passages()