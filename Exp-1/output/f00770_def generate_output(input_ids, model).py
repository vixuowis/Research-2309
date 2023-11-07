import torch

def generate_output(input_ids, model):
    """
    This function generates the output using the given input_ids and model.

    Params:
    - input_ids: A tensor containing the input ids.
    - model: The model to use for generating the output.

    Returns:
    - The output logits as a tensor.
    """
    output = model(input_ids)
    return output.logits
