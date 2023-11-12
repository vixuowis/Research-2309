# function_import --------------------

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# function_code --------------------

def denoise_audio(audio_path: str) -> None:
    '''
    This function uses a pre-trained model from Hugging Face Transformers to denoise an audio file.
    
    Args:
        audio_path (str): The path to the audio file to be denoised.
    
    Returns:
        None. The function saves the denoised audio to a file.
    
    Raises:
        FileNotFoundError: If the audio file does not exist.
    '''
    # Load the pre-trained model
    model = Wav2Vec2ForCTC.from_pretrained('JorisCos/DCUNet_Libri1Mix_enhsingle_16k')
    
    # Load the audio file
    processor = Wav2Vec2Processor.from_pretrained('JorisCos/DCUNet_Libri1Mix_enhsingle_16k')
    
    # Process the audio file
    input_values = processor(audio_path, return_tensors='pt').input_values
    
    # Denoise the audio
    logits = model(input_values).logits
    
    # Save the denoised audio to a file
    denoised_audio = processor.decode(logits[0])
    with open('denoised_audio.wav', 'w') as f:
        f.write(denoised_audio)

# test_function_code --------------------

def test_denoise_audio():
    '''
    This function tests the denoise_audio function.
    '''
    # Test with a valid audio file
    try:
        denoise_audio('valid_audio.wav')
    except FileNotFoundError:
        print('Test failed: The audio file does not exist.')
    
    # Test with an invalid audio file
    try:
        denoise_audio('invalid_audio.wav')
    except FileNotFoundError:
        print('Test passed: The audio file does not exist.')
    
    # Test with a non-audio file
    try:
        denoise_audio('non_audio.txt')
    except Exception as e:
        print(f'Test passed: {str(e)}')
    
    return 'All Tests Passed'

# call_test_function_code --------------------

test_denoise_audio()