from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import gym

# Function to load the Pong No Frameskip-v4 model
# This function uses the Stable Baselines3 library to load a pre-trained model
# The model is trained using the PPO (Proximal Policy Optimization) algorithm
# The model is loaded from the Hugging Face model hub

def load_pong_model(repo_id, filename):
    # Load the environment
    env = gym.make('PongNoFrameskip-v4')
    env = DummyVecEnv([lambda: env])

    # Load the pre-trained model from the specified repository
    model = PPO.load_from_hub(repo_id, filename)

    # Return the model and the environment
    return model, env