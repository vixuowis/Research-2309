# requirements_file --------------------

!pip install -U espnet.git@d5b5ec7b2e77bd3e10707141818b7e6c57ac6b3f tensorflow transformers 

# function_import --------------------

import subprocess
import os

# function_code --------------------

def setup_espnet():
    """
    Sets up the ESPnet environment and downloads the necessary TTS model.
    """
    # Navigate to the ESPnet directory and check out the specified commit
    os.chdir('espnet')
    result = subprocess.run(['git', 'checkout', 'd5b5ec7b2e77bd3e10707141818b7e6c57ac6b3f'], capture_output=True)
    if result.returncode != 0:
        raise Exception("Failed to checkout ESPnet commit: " + result.stderr.decode())

    # Install ESPnet package
    result = subprocess.run(['pip', 'install', '-e', '.'], capture_output=True)
    if result.returncode != 0:
        raise Exception("Failed to install ESPnet: " + result.stderr.decode())

    # Navigate to the amadeus TTS recipe directory
    os.chdir('egs2/amadeus/tts1')

    # Run the script to download the model
    result = subprocess.run(['./run.sh', '--skip_data_prep', 'false', '--skip_train', 'true', '--download_model', 'mio/amadeus'], capture_output=True)
    if result.returncode != 0:
        raise Exception("Failed to download the model: " + result.stderr.decode())

    # We assume the command also prepares the TTS functionality, hence no further action is required here


# test_function_code --------------------

def test_setup_espnet():
    print("Testing ESPnet setup.")

    try:
        setup_espnet()
        print("Setup completed successfully.")
    except Exception as e:
        print(f"Test failed with exception: {e}")
        assert False

    # Check if the model file exists
    model_path = 'egs2/amadeus/tts1/exp/tts_train_raw_phn_tacotron_g2p_en_no_space/train.loss.ave_5best.pth'
    assert os.path.isfile(model_path), "Model file does not exist."

    print("ESPnet setup test passed.")

# Run the test function
test_setup_espnet()
