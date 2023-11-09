def test_setup_tinyroberta_for_qa():
    nlp = setup_tinyroberta_for_qa()
    QA_input = {
      'question': 'Why is model conversion important?',
      'context': 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'
    }
    res = nlp(QA_input)
    assert 'answer' in res
    assert 'score' in res
    assert 'start' in res
    assert 'end' in res

test_setup_tinyroberta_for_qa()