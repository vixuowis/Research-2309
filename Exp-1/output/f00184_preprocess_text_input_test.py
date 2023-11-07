from f00184_preprocess_text_input import *
def test_preprocess_text_input():
    text = "A list of colors: red, blue"
    model_name = "mistralai/Mistral-7B-v0.1"
    padding_side = "left"
    expected_output = {
        "input_ids": tensor([[...]]),
        "attention_mask": tensor([[...]])
    }

    output = preprocess_text_input(text, model_name, padding_side)

    assert output.keys() == expected_output.keys()
    assert torch.equal(output["input_ids"], expected_output["input_ids"])
    assert torch.equal(output["attention_mask"], expected_output["attention_mask"])

    print("All test cases pass.")
