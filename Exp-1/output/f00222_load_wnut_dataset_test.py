from f00222_load_wnut_dataset import *
def test_load_wnut_dataset():
    dataset = load_wnut_dataset()
    assert len(dataset) == 3
    assert "train" in dataset
    assert "validation" in dataset
    assert "test" in dataset


def main():
    test_load_wnut_dataset()


if __name__ == "__main__":
    main()
