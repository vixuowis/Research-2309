# This is the test function for load_and_test_model
# It loads the model and tests it on the LunarLander-v2 environment
# It then asserts that the mean reward is within a certain range
# This is to account for the inherent variability in reinforcement learning

def test_load_and_test_model():
    mean_reward, std_reward = load_and_test_model()
    assert 250 <= mean_reward <= 300, 'The model performance is not within the expected range'