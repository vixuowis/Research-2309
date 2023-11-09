def test_find_relevant_passage():
    question = 'How many people live in Berlin?'
    candidate_passages = ['Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.', 'New York City is famous for the Metropolitan Museum of Art.']
    assert find_relevant_passage(question, candidate_passages) == 'Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.'

test_find_relevant_passage()