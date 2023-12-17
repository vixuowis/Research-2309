# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoModel

# function_code --------------------

def load_and_normalize_model(model_name, mean, std):
    """
    Load a pretrained Decision Transformer model and normalize its inputs
    
    Args:
        model_name (str): The name of the pretrained model to load.
        mean (list): The mean values used for normalization.
        std (list): The standard deviation values used for normalization.
    
    Returns:
        A tuple of the `AutoModel` instance and a normalization function.
        
    Raises:
        ValueError: If the mean or std lists do not match expected dimension.
    """
    if not (len(mean) == 11 and len(std) == 11):
        raise ValueError('Mean and std lists must each have 11 elements for normalization.')
    model = AutoModel.from_pretrained(model_name)
    def normalize(x):
        return [(xi - m) / s for xi, m, s in zip(x, mean, std)]
    return model, normalize

# test_function_code --------------------

def test_load_and_normalize_model():
    print("Testing started.")
    test_model_name = 'edbeeching/decision-transformer-gym-hopper-medium'
    test_mean = [1.311279, -0.08469521, -0.5382719, -0.07201576, 0.04932366, 2.1066856, -0.15017354, 0.00878345, -0.2848186, -0.18540096, -0.28461286]
    test_std = [0.17790751, 0.05444621, 0.21297139, 0.14530419, 0.6124444, 0.85174465, 1.4515252, 0.6751696, 1.536239, 1.6160746, 5.6072536]

    print("Testing case [1/3] started.")
    try:
        model, normalize = load_and_normalize_model(test_model_name, test_mean, test_std)
        assert model is not None and callable(normalize), f"Test case [1/3] failed: Model loading or normalize function is incorrect."
    except ValueError as e:
        assert False, f"Test case [1/3] failed with ValueError: {e}"
    
    print("Testing case [2/3] started.")
    input_features = [0] * 11
    normalized_features = normalize(input_features)
    assert all(isinstance(x, float) for x in normalized_features), f"Test case [2/3] failed: Normalization did not return float values."

    print("Testing case [3/3] started.")
    try:
        load_and_normalize_model(test_model_name, test_mean[:10], test_std)
        assert False, "Test case [3/3] failed: Exception for wrong mean dimensions not raised."
    except ValueError:
        pass

    print("Testing finished.")

# call_test_function_line --------------------

test_load_and_normalize_model()