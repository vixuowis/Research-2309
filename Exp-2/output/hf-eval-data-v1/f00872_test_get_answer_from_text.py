def test_get_answer_from_text():
    question = 'What is a good example of a question answering dataset?'
    context = 'Extractive Question Answering is the task of extracting an answer from a text given a question. An example of a question answering dataset is the SQuAD dataset, which is entirely based on that task.'
    answer = get_answer_from_text(question, context)
    assert isinstance(answer, str), 'The function should return a string.'
    assert 'SQuAD' in answer, 'The function should return the correct answer.'

test_get_answer_from_text()