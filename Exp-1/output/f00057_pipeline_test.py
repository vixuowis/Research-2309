from f00057_pipeline import *
def test_pipeline():
    transcriber = pipeline(model='openai/whisper-large-v2', device_map='auto')
    assert transcriber is not None

test_pipeline()
