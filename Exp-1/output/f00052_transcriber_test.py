from f00052_transcriber import *
def test_transcriber():
    audio_file = 'https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac'
    expected_output = {'text': 'I HAVE A DREAM BUT ONE DAY THIS NATION WILL RISE UP LIVE UP THE TRUE MEANING OF ITS TREES'}
    assert transcriber(audio_file) == expected_output

test_transcriber()
