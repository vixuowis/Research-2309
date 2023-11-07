from f00875_translate_audio_to_text import *
def test_translate_audio_to_text():
    model = Wav2Vec2ForCTC.load_adapter("fra")
    processor = Wav2Vec2CTCTokenizer()
    audio = torch.tensor([0.1, 0.2, 0.3])
    sampling_rate = 16_000

    transcription = translate_audio_to_text(model, processor, audio, sampling_rate)
    assert transcription == "Bonjour"

    audio = torch.tensor([-0.1, -0.2, -0.3])
    sampling_rate = 16_000

    transcription = translate_audio_to_text(model, processor, audio, sampling_rate)
    assert transcription == "Au revoir"

    audio = torch.tensor([0.5, 0.6, 0.7])
    sampling_rate = 16_000

    transcription = translate_audio_to_text(model, processor, audio, sampling_rate)
    assert transcription == "Merci"
