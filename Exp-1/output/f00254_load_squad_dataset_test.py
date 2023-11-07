from f00254_load_squad_dataset import *
def test_load_squad_dataset():
    dataset = load_squad_dataset()
    assert len(dataset) == 5000
    assert "context" in dataset[0]
    assert "question" in dataset[0]
    assert "answers" in dataset[0]
    assert "id" in dataset[0]

if __name__ == "__main__":
    test_load_squad_dataset()
