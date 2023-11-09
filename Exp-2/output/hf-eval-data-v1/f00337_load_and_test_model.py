from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# This function loads a trained PPO model and tests it on the LunarLander-v2 environment
# It uses the load_from_hub function from the huggingface_sb3 package to download the trained model checkpoint
# Then it loads the trained PPO model using PPO.load(checkpoint)
# Finally, it tests the model's performance on the LunarLander-v2 environment

def load_and_test_model():
    checkpoint = load_from_hub('araffin/ppo-LunarLander-v2', 'ppo-LunarLander-v2.zip')
    model = PPO.load(checkpoint)
    env = make_vec_env('LunarLander-v2', n_envs=1)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20, deterministic=True)
    print(f'Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}')
    return mean_reward, std_reward