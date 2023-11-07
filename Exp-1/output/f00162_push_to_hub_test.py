from f00162_push_to_hub import *
def test_push_to_hub():
    assert push_to_hub() == 'https://huggingface.co/username/model_name'
