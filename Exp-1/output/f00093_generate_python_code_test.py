from f00093_generate_python_code import *
def test_generate_python_code():
    processed_dataset = {
        "input_values": [np.ones((100000,)), np.ones((100000,))]
    }
    expected_code = """
import numpy as np

>>> processed_dataset["input_values"][0].shape
(100000,)

>>> processed_dataset["input_values"][1].shape
(100000,)
"""
    assert generate_python_code(processed_dataset) == expected_code

test_generate_python_code()
