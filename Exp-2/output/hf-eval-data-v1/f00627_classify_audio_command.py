from transformers import AutoModelForAudioClassification


def classify_audio_command(audio_file_path):
    """
    This function classifies the audio command using a pre-trained model from Hugging Face Transformers.
    The model used is 'MIT/ast-finetuned-speech-commands-v2' which is specifically trained for audio classification tasks.
    
    Parameters:
    audio_file_path (str): The path to the audio file to be classified.
    
    Returns:
    str: The classified command.
    """
    # Load the pre-trained model
    audio_classifier = AutoModelForAudioClassification.from_pretrained('MIT/ast-finetuned-speech-commands-v2')
    
    # Classify the audio command
    result = audio_classifier(audio_file_path)
    
    return result