from f00129_set_format import *
def test_set_format():
    dataset = Dataset.from_dict({"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]})
    set_format("torch")
    assert isinstance(dataset["input_ids"], torch.Tensor)
    assert isinstance(dataset["attention_mask"], torch.Tensor)

test_set_format()
