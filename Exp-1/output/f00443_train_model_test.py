from f00443_train_model import *
def test_train_model():
    id2label = {0: 'cat', 1: 'dog'}
    label2id = {'cat': 0, 'dog': 1}
    train_model(id2label, label2id)
