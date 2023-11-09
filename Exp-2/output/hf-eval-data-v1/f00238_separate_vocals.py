from transformers import pipeline

# This function separates vocals from a song using the 'Awais/Audio_Source_Separation' pre-trained model from Hugging Face Transformers.
# It takes an audio file path as input and returns an array of output audio files, where each file contains one of the separated sources (vocals, instruments, etc.).
# This is especially helpful for karaoke nights when we want only the instrumental track.
def separate_vocals(audio_file_path):
    # Create an audio source separation model
    source_separation = pipeline('audio-source-separation', model='Awais/Audio_Source_Separation')
    # Separate the audio sources
    separated_audio_sources = source_separation(audio_file_path)
    return separated_audio_sources