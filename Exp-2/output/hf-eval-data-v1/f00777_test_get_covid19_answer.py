def test_get_covid19_answer():
    question = 'What are the symptoms of COVID-19?'
    context = 'The most common symptoms of COVID-19 include fever, dry cough, and shortness of breath. Some patients may also experience fatigue, headache, and muscle pain.'
    answer = get_covid19_answer(question, context)
    assert isinstance(answer, str), 'The function should return a string.'
    assert answer == 'fever, dry cough, and shortness of breath', 'The function returned the wrong answer.'

test_get_covid19_answer()