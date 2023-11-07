from f00859_load_model_with_exllama import *
import torch

def test_load_model_with_exllama():
    username = '{your_username}'
    model = load_model_with_exllama(username)
    assert isinstance(model, torch.nn.Module)


def test_all():
    test_load_model_with_exllama()
