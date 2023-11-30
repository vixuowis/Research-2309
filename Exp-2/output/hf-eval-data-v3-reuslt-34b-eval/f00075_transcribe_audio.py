# function_import --------------------

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset

# function_code --------------------

def transcribe_audio(audio_sample):
    '''
    Transcribe audio using the openai/whisper-tiny model.

    Args:
        audio_sample (dict): A dictionary containing 'array' and 'sampling_rate' of the audio.

    Returns:
        str: The transcribed text.
    '''

     # Load processor
    processor = WhisperProcessor.from_pretrained("cacos/whisper-tiny")

    # Reload model discarding model weights, use for inference only!
    model = WhisperForConditionalGeneration.from_pretrained('facebook/wav2vec2-base-960h', from_tf=False)

    # Preprocess audio dict to torch tensor -> ValueError: Audio array not of int16 dtype
    features = processor(audio_sample, sampling_rate=16_000, return_tensors="pt", padding=True)
    
    # Generate prediction
    logits = model(features.input_values, attention_mask=features.attention_mask).logits
    idx = logits.argmax(-1).flatten().tolist()[0]
    
    # Decode transcription
    transcript = processor.decode(idx)

    return transcript

# function call --------------------

if __name__ == '__main__':

    ds_iter = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
    audio_sample = next(ds_iter)['audio']
    
    transcription = transcribe_audio(audio_sample)
    print('Transcribed text: ', transcription)

# test_function_code --------------------

def test_transcribe_audio():
    '''
    Test the transcribe_audio function.
    '''
    ds = load_dataset('hf-internal-testing/librispeech_asr_dummy', 'clean', split='validation')
    sample = ds[0]['audio']
    transcription = transcribe_audio(sample)
    assert isinstance(transcription, str), 'The result should be a string.'
    return 'All Tests Passed'


# call_test_function_code --------------------

test_transcribe_audio()