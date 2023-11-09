def test_find_most_relevant_answer():
    question = 'What is the capital of France?'
    answers = ['Paris', 'London', 'Berlin']
    assert find_most_relevant_answer(question, answers) == 'Paris'
    
    question = 'Who is the president of the United States?'
    answers = ['Joe Biden', 'Donald Trump', 'Barack Obama']
    assert find_most_relevant_answer(question, answers) == 'Joe Biden'
    
    question = 'What is the highest mountain in the world?'
    answers = ['Mount Everest', 'K2', 'Kangchenjunga']
    assert find_most_relevant_answer(question, answers) == 'Mount Everest'
    
    # Test with an empty answers list
    question = 'What is the capital of France?'
    answers = []
    try:
        find_most_relevant_answer(question, answers)
    except ValueError as e:
        assert str(e) == 'The answers list cannot be empty.'
    else:
        assert False, 'Expected a ValueError when the answers list is empty.'

test_find_most_relevant_answer()