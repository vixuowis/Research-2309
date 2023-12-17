# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def predict_pokemon_hp(input_data):
    """
    Predict the HP of a Pokemon given its attributes.

    Parameters:
    input_data (dict): A dictionary containing the attributes of the Pokemon.

    Returns:
    float: The predicted HP of the Pokemon.
    """
    # Create a regression pipeline using the specified model
    hp_predictor = pipeline('regression', model='julien-c/pokemon-predict-hp')
    # Predict the HP using the provided input data
    predicted_hp = hp_predictor(input_data)[0]['score']
    return predicted_hp

# test_function_code --------------------

from datasets import load_dataset

def test_predict_pokemon_hp():
    print("Testing predict_pokemon_hp function started.")
    # Placeholder for obtaining sample data
    sample_data = {
        'stat_total': 300,
        'legendary': False,
        'generation': 1,
        'height_m': 0.7,
        'weight_kg': 6.9,
        'base_egg_steps': 5120
    }
    
    prediction = predict_pokemon_hp(sample_data)
    assert isinstance(prediction, float), f"Test case [1/3] failed: Expected float, got {type(prediction)}"
    print("Test case [1/3] passed.")

    # More test cases should be included here, using actual dataset sample if available

    print("Testing predict_pokemon_hp function finished.")

# Run the test function
test_predict_pokemon_hp()