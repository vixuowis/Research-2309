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
    
    processor = WhisperProcessor().from_pretrained("facebook/whisper-large-10m-en-de")
    model = WhisperForConditionalGeneration(processor.feature_extractor, 
                                            processor.tokenizer, 
                                            processor.config).from_pretrained("facebook/whisper-large-10m-en-de")
    
    input_audio = processor(audio=audio_sample['array'], sampling_rate=audio_sample['sampling_rate']).input_values

    generated_text = model.generate(input_ids=input_audio)

    text = processor.batch_decode(generated_text, skip_special_tokens=True)[0]

    return text

# function_import --------------------

dataset = load_dataset("common_voice", "fr")

dataset['train'].map(lambda audio: {"transcription": transcribe_audio({'array': audio["audio"], 'sampling_rate': 16000})})

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