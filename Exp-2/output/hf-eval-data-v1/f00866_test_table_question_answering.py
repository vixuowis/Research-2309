def test_table_question_answering():
    """
    This function tests the table_question_answering function by using a sample table and a set of questions.
    """
    import pandas as pd

    # Sample table
    data = {
        'Month': ['January', 'February', 'March', 'April'],
        'Salesperson': ['John', 'John', 'Ana', 'Ana'],
        'Sales': [1000, 1500, 2000, 2500]
    }
    table = pd.DataFrame(data)

    # Sample questions
    questions = ['What is the total sales for John?', 'Who made the highest sales in April?']

    # Expected answers
    expected_answers = ['2500', 'Ana']

    for i, question in enumerate(questions):
        assert table_question_answering(table, question) == expected_answers[i]