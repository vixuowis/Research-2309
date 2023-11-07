from f00256_get_answer_start import *
def test_get_answer_start():
    answers = {'answer_start': [515]}
    assert get_answer_start(answers) == [515]

    answers = {'answer_start': [100, 200, 300]}
    assert get_answer_start(answers) == [100, 200, 300]

    answers = {'answer_start': []}
    assert get_answer_start(answers) == []

    answers = {'answer_start': [0, 1, 2, 3, 4]}
    assert get_answer_start(answers) == [0, 1, 2, 3, 4]

    answers = {'answer_start': [10, 20, 30, 40, 50]}
    assert get_answer_start(answers) == [10, 20, 30, 40, 50]

test_get_answer_start()
