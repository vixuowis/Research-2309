from f00460_prepare_dataset import *
batch = {
    'audio': {
        'array': audio_array,
        'sampling_rate': 16000
    },
    'transcription': 'Hello, how are you?'
}

output_batch = prepare_dataset(batch)

assert 'input_values' in output_batch
assert 'input_length' in output_batch
assert len(output_batch['input_values']) == 1
assert len(output_batch['input_values'][0]) == output_batch['input_length']
