from f00452_load_minds_dataset import *
def test_load_minds_dataset():
    minds = load_minds_dataset()
    assert len(minds) == 100
    assert isinstance(minds[0], dict)
    assert "audio" in minds[0]


def test_entry():
    test_load_minds_dataset()
