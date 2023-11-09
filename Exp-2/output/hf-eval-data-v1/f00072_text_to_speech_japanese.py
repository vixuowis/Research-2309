from transformers import pipeline
import soundfile as sf

def text_to_speech_japanese(japanese_text: str, output_file: str = 'output.wav') -> None:
    '''
    This function converts a given Japanese text into a speech audio file.
    
    Args:
    japanese_text (str): The Japanese text to be converted into speech.
    output_file (str): The name of the output audio file. Default is 'output.wav'.
    
    Returns:
    None
    '''
    # Create a text-to-speech pipeline with the specified model
    tts = pipeline('text-to-speech', model='espnet/kan_bayashi_jvs_tts_finetune_jvs001_jsut_vits_raw_phn_jaconv_pyopenjta-truncated-178804')
    # Convert the Japanese text into an audio waveform
    audio_waveform = tts(japanese_text)[0]['generated_sequence']
    # Save the audio waveform to an audio file
    sf.write(output_file, audio_waveform, samplerate=24000)