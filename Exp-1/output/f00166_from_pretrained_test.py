from f00166_from_pretrained import *
def test_from_pretrained():
    model = from_pretrained("your_username/my-awesome-model")
    assert isinstance(model, AutoModel)


test_from_pretrained()
