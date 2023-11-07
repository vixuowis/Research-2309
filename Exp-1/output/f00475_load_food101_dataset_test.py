from f00475_load_food101_dataset import *
def test_load_food101_dataset():
    food = load_food101_dataset()
    assert len(food) == 5000
    assert "image" in food[0]
    assert "label" in food[0]
    assert "id" in food[0]
    assert "file" in food[0]
    assert "image/filename" in food[0]
    assert "label/coarse" in food[0]
    assert "label/fine" in food[0]
    assert "label/str" in food[0]
