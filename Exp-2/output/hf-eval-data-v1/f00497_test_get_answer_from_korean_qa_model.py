def test_get_answer_from_korean_qa_model():
    question = '서울의 수도는 무엇인가요?' # 'What is the capital of Seoul?'
    context = '서울은 대한민국의 수도입니다.' # 'Seoul is the capital of South Korea.'
    
    # The expected answer is '대한민국'
    expected_answer = '대한민국'
    
    # Use the function to get the answer
    answer = get_answer_from_korean_qa_model(question, context)
    
    # Check if the answer is correct
    assert answer == expected_answer, f'Expected {expected_answer}, but got {answer}'
    
    print('All tests passed.')

test_get_answer_from_korean_qa_model()