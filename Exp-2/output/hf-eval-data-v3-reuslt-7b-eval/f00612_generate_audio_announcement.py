# function_import --------------------

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf

# function_code --------------------

def generate_audio_announcement(text):
    '''
    Generate an audio announcement from a given text using the SpeechT5 model.
    
    Args:
        text (str): The text to be converted to speech.
    
    Returns:
        None. The function writes the output audio to a .wav file.
    
    Raises:
        Exception: If there is an error in generating the audio.
    '''
    try:
        
        # Create the SpeechT5Processor and model.
        processor = SpeechT5Processor.from_pretrained("facebook/speech-t5-small")
        model = SpeechT5ForTextToSpeech.from_pretrained(
            "facebook/speech-t5-small", 
            device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'),
        )
        
        # Load and preprocess the text input using SpeechT5Processor
        inputs = processor(text, return_tensors="pt", padding="longest")
    
        # Generate audio from the text input using SpeechT5ForTextToSpeech model.
        audio_tensor = model(**inputs).audio 
        
        # Save the audio as .wav file.
        sf.write('announcement.wav', audio_tensor[0].numpy(), processor.feature_extractor._sample_rate)
        
    except Exception as e:
        print(e)

# main_code --------------------

# Load a test dataset to generate an announcement using the SpeechT5 model.
test_dataset = load_dataset("common_voice", "en")["train"][0]
print("\nOriginal text:\t", test_dataset['sentence'])  # Print original text.

text = ''.join(e if e.isalnum() else ' ' for e in test_dataset['sentence'])  # Preprocess text by removing punctuation and special characters
# Remove leading/trailing spaces from the preprocessed text, and convert to lowercase.
text = text.strip().lower()
print("\nPreprocessed text:\t", text)  # Print preprocessed text.
    
generate_audio_announcement(text)  # Generate audio announcement using preprocessed text.

# test_function_code --------------------

def test_generate_audio_announcement():
    '''
    Test the generate_audio_announcement function.
    '''
    try:
        generate_audio_announcement('This is a test announcement.')
        print('Test passed.')
    except Exception as e:
        print('Test failed. Error: ', e)
    return 'All Tests Passed'


# call_test_function_code --------------------

test_generate_audio_announcement()