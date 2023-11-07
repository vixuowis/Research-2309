from f00231_load import *
def test_load():
    assert load('seqeval') == evaluate.load('seqeval')
    assert load('metric1') == evaluate.load('metric1')
    assert load('metric2') == evaluate.load('metric2')
    assert load('metric3') == evaluate.load('metric3')
    assert load('metric4') == evaluate.load('metric4')


test_load()
