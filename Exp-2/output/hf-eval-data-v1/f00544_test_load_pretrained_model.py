# Test function for load_pretrained_model
# The function loads a pre-trained model using the load_pretrained_model function
# It then checks if the loaded model is an instance of the PPO class
# This is done using the isinstance function
# If the loaded model is not an instance of the PPO class, the test fails

import unittest

class TestLoadPretrainedModel(unittest.TestCase):
    def test_load_pretrained_model(self):
        model = load_pretrained_model('{MODEL FILENAME}.zip')
        self.assertIsInstance(model, PPO)

# Run the test
unittest.main()