def load_and_evaluate_model(checkpoint: str, kwargs: dict, env_name: str, n_eval_episodes: int):
    """
    Load a pre-trained DQN model from a checkpoint and evaluate its performance.

    Args:
        checkpoint (str): The path to the model checkpoint.
        kwargs (dict): Additional arguments to pass to the DQN.load function.
        env_name (str): The name of the environment to create for evaluation.
        n_eval_episodes (int): The number of evaluation episodes to run.

    Returns:
        tuple: A tuple containing the mean reward and standard deviation.
    """
    from huggingface_sb3 import load_from_hub
    from stable_baselines3 import DQN
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.evaluation import evaluate_policy

    model = DQN.load(checkpoint, **kwargs)
    env = make_vec_env(env_name, n_envs=1)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, deterministic=True)
    return mean_reward, std_reward