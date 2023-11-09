def test_extract_information():
    context = 'Manuel Romero a travaillé dur dans le dépôt hugginface/transformers récemment'
    question = 'Qui a travaillé dur pour hugginface/transformers récemment?'
    answer = extract_information(context, question)
    assert 'answer' in answer
    assert isinstance(answer['answer'], str)

test_extract_information()