# requirements_file --------------------

!pip install -U transformers, torch

# function_import --------------------

from transformers import AutoModel

# function_code --------------------

def load_and_evaluate_graphormer(sample_data):
    """
    Load the pretrained Graphormer model and evaluate its performance on a sample molecular graph.
    
    Args:
    - sample_data: A molecular graph data sample on which to evaluate the Graphormer model.
    
    Returns:
    - A tuple containing the input sample data and the predicted properties.
    """
    # Load the pretrained Graphormer model
    model = AutoModel.from_pretrained('graphormer-base-pcqm4mv1')
    
    # Evaluate the model on the provided sample data
    model.eval()  # Put the model in evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        predictions = model(sample_data)
    
    return sample_data, predictions

# test_function_code --------------------

import torch

def test_load_and_evaluate_graphormer():
    print("Testing started.")
    
    # Create a sample molecular graph data (dummy data for demonstration)
    sample_data = torch.randn((1, 128))  # Assume the Graphormer model expects 128 features

    # Test case: Load model and perform inference
    print("Testing case [1/1] started.")
    try:
        _, predictions = load_and_evaluate_graphormer(sample_data)
        assert isinstance(predictions, torch.Tensor), "Predictions should be a PyTorch tensor."
        print("Testing case [1/1] passed.")
    except Exception as e:
        print(f"Test case [1/1] failed: {e}")

    print("Testing finished.")

# Run the test function
if __name__ == "__main__":
    test_load_and_evaluate_graphormer()