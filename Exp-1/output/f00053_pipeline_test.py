from f00053_pipeline import *
def test_transcriber():
    transcriber = pipeline(model='openai/whisper-large-v2')
    result = transcriber('https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac')
    assert result == {'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.'}


if __name__ == '__main__':
    test_transcriber()
