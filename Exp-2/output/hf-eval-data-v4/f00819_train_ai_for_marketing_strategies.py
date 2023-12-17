# requirements_file --------------------

!pip install -U numpy stable-baselines3 

# function_import --------------------

import numpy as np
from stable_baselines3 import DQN
from typing import Tuple
from reinforcement_learning_env import MarketingStrategiesEnv


# function_code --------------------

def train_ai_for_marketing_strategies() -> DQN:
    """
    Trains a DQN agent to find the best marketing strategies for a website by experimenting with different headlines and image combinations.
    
    Returns:
        Trained DQN model for optimizing marketing strategies.
    """
    # Create the custom environment
    env = MarketingStrategiesEnv()
    
    # Initialize the agent
    model = DQN('MlpPolicy', env, verbose=1)
    
    # Train the agent
    model.learn(total_timesteps=10000)
    
    # Save the model (optional)
    model.save("marketing_strategies_model.zip")
    
    return model


# test_function_code --------------------

def test_marketing_strategies_ai_training():
    print("Testing started.")

    # 由于是自定义环境，您需要先实现您的环境
    # 假设您的环境叫 MarketingStrategiesEnv 并已经实现了
    env = MarketingStrategiesEnv()

    # 运行测试训练
    model = train_ai_for_marketing_strategies()

    # 测试用例 1：检查是否正确初始化
    print("Testing case [1/3] started.")
    assert model is not None, f"Test case [1/3] failed: The model is not initialized."
    
    # 测试用例 2：检查模型是否能在环境中正常执行动作
    print("Testing case [2/3] started.")
    action, _ = model.predict(env.reset())
    new_state, reward, done, _ = env.step(action)
    assert not done, f"Test case [2/3] failed: Environment finished prematurely after a single step."

    # 测试用例 3：检查模型保存功能
    print("Testing case [3/3] started.")
    try:
        model.save("test_marketing_strategies_model.zip")
        model = DQN.load("test_marketing_strategies_model.zip")
        assert model is not None, "Test case [3/3] failed: Model not saved or loaded properly."
    except IOError:
        assert False, "Test case [3/3] failed: Saving or loading model caused IOError."

    print("Testing finished.")

# 运行测试函数
test_marketing_strategies_ai_training()
