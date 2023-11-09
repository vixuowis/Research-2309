# function_import --------------------

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# function_code --------------------

def transcribe_audio(audio_filepath):
    """
    Transcribes an audio file using the Wav2Vec2ForCTC model from Hugging Face Transformers.

    Args:
        audio_filepath (str): The path to the audio file to transcribe.

    Returns:
        str: The transcription of the audio file.
    """
    model = Wav2Vec2ForCTC.from_pretrained('jbetker/wav2vec2-large-robust-ft-libritts-voxpopuli')
    processor = Wav2Vec2Processor.from_pretrained('jbetker/wav2vec2-large-robust-ft-libritts-voxpopuli')

    inputs = processor(audio_filepath, return_tensors="pt", padding=True)
    outputs = model(inputs.input_values.to("cuda"), attention_mask=inputs.attention_mask.to("cuda"), labels=inputs.labels.to("cuda"))

    transcription = processor.decode(outputs.logits.argmax(dim=-1)[0])

    return transcription

# test_function_code --------------------

def test_transcribe_audio():
    """
    Tests the transcribe_audio function.

    Raises:
        AssertionError: If the function does not return a string.
    """
    transcription = transcribe_audio('test_audio.wav')
    assert isinstance(transcription, str), 'The transcription should be a string.'

# call_test_function_code --------------------

test_transcribe_audio()