def test_extract_financial_info():
    '''
    This function tests the 'extract_financial_info' function.
    It uses a sample table and a set of queries, and checks if the function returns the correct answers.
    '''
    data = {'year': [1896, 1900, 1904, 2004, 2008, 2012], 'city': ['Athens', 'Paris', 'St. Louis', 'Athens', 'Beijing', 'London']}
    table = pd.DataFrame.from_dict(data)
    queries = ['In which year did Beijing host the Olympic Games?', 'Which city hosted the Olympic Games in 1900?']
    expected_answers = ['2008', 'Paris']
    for query, expected_answer in zip(queries, expected_answers):
        assert extract_financial_info(table, query) == expected_answer

test_extract_financial_info()