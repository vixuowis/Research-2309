def test_get_answer():
    question = 'What is the capital of Germany?'
    context = 'Berlin is the capital of Germany.'
    assert get_answer(question, context) == 'Berlin'

test_get_answer()