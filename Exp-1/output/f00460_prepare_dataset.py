from typing import *
from transformers import Wav2Vec2Processor

processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-large-960h')

def prepare_dataset(batch):
    # Calls the audio column to load and resample the audio file.
    # Extracts the input_values from the audio file and tokenize the transcription column with the processor.
    audio = batch['audio']
    batch = processor(audio['array'], sampling_rate=audio['sampling_rate'], text=batch['transcription'])
    batch['input_length'] = len(batch['input_values'][0])
    return batch
