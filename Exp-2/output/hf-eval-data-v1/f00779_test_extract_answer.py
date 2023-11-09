def test_extract_answer():
    question = 'What is the penalty for breaking the contract?'
    context = 'The penalty for breaking the contract is generally...'
    answer = extract_answer(question, context)
    assert isinstance(answer, str), 'The function should return a string.'
    assert answer != '', 'The function should return a non-empty string.'

test_extract_answer()