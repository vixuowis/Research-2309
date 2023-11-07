from f00826_transcriber import *
def test_transcriber():
    transcriber = pipeline(task='automatic-speech-recognition', model='openai/whisper-small')
    test_cases = [
        ('https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac', {'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.'}),
        # Add more test cases here
    ]

    for url, expected_output in test_cases:
        result = transcriber(url)
        assert result == expected_output

    print('All test cases pass')

test_transcriber()
