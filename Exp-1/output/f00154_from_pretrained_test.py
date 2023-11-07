from f00154_from_pretrained import *
def test_from_pretrained():
    model = from_pretrained('julien-c/EsperBERTo-small', revision='v2.0.1')
    assert isinstance(model, AutoModel)

    # rest of the test cases

