from f00002_classifier import *

def test_classifier():
    assert classifier("We are very happy to show you the 🤗 Transformers library.") == [{'label': 'POSITIVE', 'score': 0.9998}]
    assert classifier("I am feeling sad today.") == [{'label': 'NEGATIVE', 'score': 0.9997}]
    assert classifier("This movie is so boring.") == [{'label': 'NEGATIVE', 'score': 0.9999}]
    assert classifier("I love this song!") == [{'label': 'POSITIVE', 'score': 0.9999}]
    assert classifier("The weather is perfect today.") == [{'label': 'POSITIVE', 'score': 0.9999}]

test_classifier()
