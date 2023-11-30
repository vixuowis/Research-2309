# function_import --------------------

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# function_code --------------------

def generate_dialogue(user_input):
    """
    Generate a dialogue response using DialoGPT-large model.

    Args:
        user_input (str): The user's input to which the chatbot should respond.

    Returns:
        str: The chatbot's response.

    Raises:
        OSError: If there is an issue with loading the pre-trained model or tokenizer.
    """

    try:
        # Load pre-trained DialoGPT-large model and tokenizer
        dialogue_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")
        dialogue_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
    except OSError:
        # Handle error if pre-trained model and tokenizer files do not exist locally.
        print(f"\n\nPre-trained DialoGPT-large model and tokenizer not found.\n")
        
        # Prompt user to download pre-trained model and tokenizer or to quit the program
        while True:
            download_or_quit = input("Would you like to download these files now? ('y' for yes, 'q' to quit)  ")
            
            if download_or_quit == "y":
                print("\nDownloading pre-trained DialoGPT-large model and tokenizer...\n")
                
                # Download and extract the pre-trained DialoGPT-large model
                os.system("wget https://cdn.huggingface.co/microsoft/DialoGPT-large/pytorch_model.bin -P ./models")
                os.system("unzip pytorch_model.bin -d ./models")
                
                # Download and extract the pre-trained DialoGPT-large tokenizer
                os.system("wget https://cdn.huggingface.co/microsoft/DialoGPT-large/vocab.json -P ./tokenizers")
                os.system("unzip vocab.json -d ./tokenizers")
                
                print("\nDownload and extraction complete!\n")
                
                break # Break out of the while loop
            elif download_or_quit == "q":
                quit() # Quit the program if user does not want to download
            else:
                print("Please enter 'y' or 'q'.\n")
        
        # Load pre-trained DialoGPT-large model and tokenizer
       

# test_function_code --------------------

def test_generate_dialogue():
    """
    Test the generate_dialogue function.
    """
    test_input = 'How do I search for scientific papers?'
    response = generate_dialogue(test_input)
    assert isinstance(response, str), 'The response should be a string.'

    test_input = 'What is the weather like today?'
    response = generate_dialogue(test_input)
    assert isinstance(response, str), 'The response should be a string.'

    test_input = 'Tell me a joke.'
    response = generate_dialogue(test_input)
    assert isinstance(response, str), 'The response should be a string.'

    return 'All Tests Passed'


# call_test_function_code --------------------

test_generate_dialogue()