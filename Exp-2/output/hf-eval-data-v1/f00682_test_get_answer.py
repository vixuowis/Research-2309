# Test function for get_answer
# This function uses assert to verify the correctness of the get_answer function
# It uses a sample question and context from the SQuAD 2.0 dataset for testing

def test_get_answer():
    question = 'What was the main cause of the war?'
    context = 'World War I was primarily caused by a complex web of factors including political, economic, and social issues. However, the assassination of Archduke Franz Ferdinand of Austria is often cited as the immediate trigger for the conflict.'
    answer = get_answer(question, context)
    assert 'assassination of Archduke Franz Ferdinand of Austria' in answer, 'Test failed!'
    print('Test passed!')

test_get_answer()