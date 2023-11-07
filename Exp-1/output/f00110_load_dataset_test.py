from f00110_load_dataset import *
def test_load_dataset():
    dataset = load_dataset("yelp_review_full")
    assert len(dataset["train"]) > 0
    assert len(dataset["train"][0]) == 2

test_load_dataset()
