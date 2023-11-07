from f00674_generate_speech import *
def test_generate_speech():
    vocoder = Vocoder()
    spectrogram = torch.randn(1, 80, 100)
    speech = generate_speech(vocoder, spectrogram)
    assert isinstance(speech, torch.Tensor)
    assert speech.shape == (1, 10000)
    
    spectrogram = torch.randn(5, 80, 200)
    speech = generate_speech(vocoder, spectrogram)
    assert isinstance(speech, torch.Tensor)
    assert speech.shape == (5, 20000)
    
    spectrogram = torch.randn(10, 80, 300)
    speech = generate_speech(vocoder, spectrogram)
    assert isinstance(speech, torch.Tensor)
    assert speech.shape == (10, 30000)
    
    print('All test cases pass')
    
test_generate_speech()
