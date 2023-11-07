from f00642_preprocess_data import *
def test_preprocess_data():
    checkpoint = 'microsoft/speecht5_tts'
    processor = preprocess_data(checkpoint)
    assert isinstance(processor, SpeechT5Processor)


test_preprocess_data()
