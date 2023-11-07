from f00108_prepare_dataset import *
import torch


def test_prepare_dataset():
    example = {
        "audio": {
            "array": torch.randn(16000),
        },
        "text": "This is a test sentence",
    }

    processed_example = prepare_dataset(example)

    assert "input_values" in processed_example
    assert "labels" in processed_example


test_prepare_dataset()
