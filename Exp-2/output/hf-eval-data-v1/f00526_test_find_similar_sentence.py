def test_find_similar_sentence():
    source_sentence = '我爱吃苹果'
    sentences_to_compare = ['我喜欢吃香蕉', '我爱吃橙子', '我爱吃苹果']
    assert find_similar_sentence(source_sentence, sentences_to_compare) == '我爱吃苹果'
    
    source_sentence = '今天天气很好'
    sentences_to_compare = ['今天天气不错', '我喜欢今天的天气', '今天是个好天气']
    assert find_similar_sentence(source_sentence, sentences_to_compare) == '今天是个好天气'

test_find_similar_sentence()