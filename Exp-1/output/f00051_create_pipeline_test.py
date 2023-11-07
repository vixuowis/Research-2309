from f00051_create_pipeline import *
def test_create_pipeline():
    transcriber = create_pipeline('automatic-speech-recognition')
    assert isinstance(transcriber, Pipeline)


test_create_pipeline()
