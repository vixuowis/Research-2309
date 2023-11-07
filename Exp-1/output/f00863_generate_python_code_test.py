from f00863_generate_python_code import *
def test_generate_python_code():
    model = RwkvModel.from_pretrained("sgugger/rwkv-430M-pile")
    tokenizer = AutoTokenizer.from_pretrained("sgugger/rwkv-430M-pile")

    inputs = tokenizer("This is an example.", return_tensors="pt")

    assert isinstance(inputs, torch.Tensor)
    assert inputs.shape == (1, 7)
