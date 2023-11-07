from f00607_subfinder import *
def test_subfinder():
    words_list = ['This', 'function', 'takes', 'two', 'lists', 'as', 'input', ',', 'words_list', 'and', 'answer_list', '.']
    answer_list = ['two', 'lists', 'as', 'input']
    assert subfinder(words_list, answer_list) == (answer_list, 3, 6)
    answer_list = ['answer_list', '.']
    assert subfinder(words_list, answer_list) == (answer_list, 10, 11)
    answer_list = ['no', 'match']
    assert subfinder(words_list, answer_list) == (None, 0, 0)

test_subfinder()
