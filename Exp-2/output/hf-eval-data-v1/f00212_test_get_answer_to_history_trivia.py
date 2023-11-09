def test_get_answer_to_history_trivia():
    # Define a context and a question for testing
    context = 'In 1492, Christopher Columbus sailed the ocean blue, discovering the New World.'
    question = 'Who discovered the New World?'
    
    # Get the answer to the question
    answer = get_answer_to_history_trivia(context, question)
    
    # Assert that the answer is correct
    assert answer == 'Christopher Columbus', f'Expected Christopher Columbus, but got {answer}'
    
    # Define another context and question for testing
    context = 'The Great Wall of China is one of the greatest sights in the world — the longest wall in the world, an awe-inspiring feat of ancient defensive architecture.'
    question = 'What is the longest wall in the world?'
    
    # Get the answer to the question
    answer = get_answer_to_history_trivia(context, question)
    
    # Assert that the answer is correct
    assert answer == 'The Great Wall of China', f'Expected The Great Wall of China, but got {answer}'

test_get_answer_to_history_trivia()