from f00192_load_imdb_dataset import *
def test_load_imdb_dataset():
    imdb = load_imdb_dataset()
    assert isinstance(imdb, dict)
    assert "train" in imdb
    assert "test" in imdb
    assert imdb["train"] is not None
    assert imdb["test"] is not None
