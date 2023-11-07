from typing import *
import numpy as np

def generate_python_code(processed_dataset):
    """Generate python code based on the processed dataset.

    Args:
        processed_dataset (dict): The processed dataset containing input values.

    Returns:
        str: The generated python code.
    """
    code = """
    import numpy as np

    >>> processed_dataset["input_values"][0].shape
    (100000,)

    >>> processed_dataset["input_values"][1].shape
    (100000,)
    """
    return code
