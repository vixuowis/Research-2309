from transformers import AutoModelForAudioToAudio
from asteroid import AudioFileProcessor


def enhance_audio(input_audio_path: str, output_audio_path: str):
    """
    This function enhances a single audio track, possibly containing dialogue, music and background noise, extracted from a video game.
    It uses the pretrained model 'JorisCos/DCCRNet_Libri1Mix_enhsingle_16k' from Hugging Face Transformers.
    The model is specifically trained for enhancing single audio tracks and reducing noise.
    
    Parameters:
    input_audio_path (str): The path to the input audio file in wav format.
    output_audio_path (str): The path where the enhanced audio file will be saved in wav format.
    """
    # Load the pretrained model
    audio_to_audio_model = AutoModelForAudioToAudio.from_pretrained('JorisCos/DCCRNet_Libri1Mix_enhsingle_16k')
    # Create an AudioFileProcessor object
    processor = AudioFileProcessor(audio_to_audio_model)
    # Process the input audio file and save the enhanced audio
    processor.process_file(input_audio_path, output_audio_path)