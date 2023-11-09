def test_table_based_question_answering():
    # Test data
    data = {
        'year': [1896, 1900, 1904, 2004, 2008, 2012],
        'city': ['athens', 'paris', 'st. louis', 'athens', 'beijing', 'london']
    }

    # Test query
    query = 'In which year did beijing host the Olympic Games?'

    # Expected answer
    expected_answer = '2008'

    # Get the function's answer
    answer = table_based_question_answering(data, query)

    # Assert that the function's answer is close to the expected answer
    assert answer == expected_answer, f'Expected {expected_answer}, but got {answer}'

    print('All tests passed.')

test_table_based_question_answering()