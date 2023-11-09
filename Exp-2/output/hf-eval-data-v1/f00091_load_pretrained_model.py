import rl_zoo3
from stable_baselines3 import PPO

# Function to load a pre-trained model from the RL Zoo
# The function takes the filename of the model as an argument
# The model is a PPO agent trained on the seals/CartPole-v0 environment
# This model can be used to optimize warehouse loading and unloading tasks

def load_pretrained_model(filename):
    model = rl_zoo3.load_from_hub(repo_id='HumanCompatibleAI/ppo-seals-CartPole-v0', filename=filename)
    return model