from f00275_get_highest_probability import *
import torch


def test_get_highest_probability():
    outputs = torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.6, 0.3]])
    result = get_highest_probability(outputs)
    assert result == (2, 1)

    outputs = torch.tensor([[0.1, 0.2, 0.7], [0.4, 0.3, 0.3]])
    result = get_highest_probability(outputs)
    assert result == (2, 0)

    outputs = torch.tensor([[0.1, 0.2, 0.7], [0.4, 0.3, 0.3], [0.2, 0.1, 0.7]])
    result = get_highest_probability(outputs)
    assert result == (2, 2)


test_get_highest_probability()
