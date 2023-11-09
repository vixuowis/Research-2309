from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch


def transcribe_audio(audio_data):
    """
    Transcribes audio data into text using the Wav2Vec2 model from Hugging Face Transformers.

    Args:
        audio_data (np.array): The audio data to transcribe.

    Returns:
        str: The transcribed text.
    """
    processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-large-960h-lv60-self')
    model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-large-960h-lv60-self')

    input_values = processor(audio_data, return_tensors='pt', padding='longest').input_values

    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)

    return transcription