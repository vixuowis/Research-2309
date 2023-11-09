# Test function for load_pretrained_model
# The function loads a pre-trained model and checks if it is an instance of the PPO class
# The test does not compare numbers strictly and uses the assert keyword

def test_load_pretrained_model():
    model = load_pretrained_model('{MODEL FILENAME}.zip')
    assert isinstance(model, PPO), 'Model loading failed'