from typing import *
import torch


def transcribe_audio(audio_data, model, processor):
    """Transcribes the audio data using the provided model and processor.

    Args:
        audio_data (torch.Tensor): The audio data to transcribe.
        model (Wav2Vec2ForCTC): The pre-trained model.
        processor (Wav2Vec2Processor): The processor used for encoding and decoding.

    Returns:
        str: The transcribed text.
    """
    inputs = processor(audio_data, sampling_rate=16_000, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**inputs).logits

    ids = torch.argmax(outputs, dim=-1)[0]
    transcription = processor.decode(ids)

    return transcription

