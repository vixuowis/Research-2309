def test_get_medical_answer():
    """
    This function tests the get_medical_answer function.
    It uses a sample context and question and checks if the function returns a non-empty string as answer.
    """
    context = 'Coronavirus disease (COVID-19) is an infectious disease caused by a newly discovered coronavirus.'
    question = 'What causes COVID-19?'
    answer = get_medical_answer(context, question)
    assert isinstance(answer, str), 'Answer must be a string'
    assert len(answer) > 0, 'Answer cannot be an empty string'

test_get_medical_answer()