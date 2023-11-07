from f00707_from_pretrained import *
def test_from_pretrained():
    assert from_pretrained('distilbert-base-uncased') == tf_model
    assert from_pretrained('distilbert-base-uncased', my_config) == tf_model
    assert from_pretrained('distilbert-base-uncased', my_config, input1, input2) == tf_model
    assert from_pretrained('distilbert-base-uncased', config=my_config) == tf_model
    assert from_pretrained('distilbert-base-uncased', config=my_config, input1=input1, input2=input2) == tf_model

test_from_pretrained()
