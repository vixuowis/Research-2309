from typing import *
import torch


def translate_audio_to_text(model, processor, audio, sampling_rate):
    """Translate audio to text using the given model and processor.

    Args:
        model (Wav2Vec2ForCTC): The pre-trained model.
        processor (Wav2Vec2Processor): The processor used for the model.
        audio (torch.Tensor): The audio data.
        sampling_rate (int): The sampling rate of the audio data.

    Returns:
        str: The transcribed text.
    """
    inputs = processor(audio, sampling_rate=sampling_rate, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**inputs).logits

    ids = torch.argmax(outputs, dim=-1)[0]
    transcription = processor.decode(ids)

    return transcription
