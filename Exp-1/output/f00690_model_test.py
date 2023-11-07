from f00690_model import *
def test_model():
    inputs = torch.tensor([[1, 2, 3], [4, 5, 6]])
    langs = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    expected_outputs = torch.tensor([[0.5, 0.6], [0.7, 0.8]])
    outputs = model(inputs, langs)
    assert torch.allclose(outputs, expected_outputs)

test_model()
