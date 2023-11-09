import os

# Function to test the train_soccer_agent function
# @param: None
# @return: None

def test_train_soccer_agent() -> None:
    # Define a test configuration file and run id
    test_config_file = "test_config.yaml"
    test_run_id = "test_run"
    
    # Create a test configuration file
    with open(test_config_file, "w") as file:
        file.write("""
        behaviors:
          SoccerTwos:
            trainer_type: poca
            hyperparameters:
              learning_rate: 0.0003
              batch_size: 1024
              buffer_size: 10240
            network_settings:
              hidden_units: 128
              num_layers: 2
        """)
    
    # Test the train_soccer_agent function
    try:
        train_soccer_agent(test_config_file, test_run_id)
    except Exception as e:
        assert False, f"The function failed with error: {str(e)}"
    
    # Delete the test configuration file
    os.remove(test_config_file)