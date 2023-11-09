import rl_zoo3
from stable_baselines3 import PPO

# Function to load a pre-trained model from the RL Zoo
# The function takes the filename of the model as an argument
# It uses the load_from_hub function provided by the RL Zoo to load the model
# The loaded model is a PPO agent trained on the CartPole-v1 environment
# The function returns the loaded model

def load_pretrained_model(model_filename):
    ppo = rl_zoo3.load_from_hub(repo_id='sb3/ppo-CartPole-v1', filename=model_filename)
    return ppo