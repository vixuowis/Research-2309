from f00056_pipeline import *
def test_pipeline():
    transcriber = pipeline(model='openai/whisper-large-v2', device=0)

    # Test case 1
    speech = 'Hello, how are you?'
    transcription = transcriber(speech)
    assert transcription == 'Hello, how are you?'

    # Test case 2
    speech = 'I am doing great!'
    transcription = transcriber(speech)
    assert transcription == 'I am doing great!'

    # Test case 3
    speech = 'What are you up to?'
    transcription = transcriber(speech)
    assert transcription == 'What are you up to?'

    # Test case 4
    speech = 'I am just working on some projects.'
    transcription = transcriber(speech)
    assert transcription == 'I am just working on some projects.'

    # Test case 5
    speech = 'That sounds interesting. Good luck!'
    transcription = transcriber(speech)
    assert transcription == 'That sounds interesting. Good luck!'

test_pipeline()
