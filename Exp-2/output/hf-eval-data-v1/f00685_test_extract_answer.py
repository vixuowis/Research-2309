def test_extract_answer():
    '''
    This function tests the extract_answer function.
    It uses a sample question and context, and checks if the extracted answer is as expected.
    '''
    question = 'What are the benefits of exercise?'
    context = 'Exercise helps maintain a healthy body weight, improves cardiovascular health, and boosts the immune system.'
    expected_answer = 'maintain a healthy body weight, improves cardiovascular health, and boosts the immune system'
    assert extract_answer(question, context) == expected_answer