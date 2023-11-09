def test_load_pong_model():
    # Define the repository ID and the filename of the model
    repo_id = 'sb3/ppo-PongNoFrameskip-v4'
    filename = '{MODEL FILENAME}.zip'

    # Load the model and the environment
    model, env = load_pong_model(repo_id, filename)

    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

    # Check if the model's performance is within the expected range
    assert 20 <= mean_reward <= 22, f'Expected mean reward to be within 20 and 22, but got {mean_reward}'