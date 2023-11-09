def test_classify_emotion():
    """
    Test the classify_emotion function with some example messages.
    """
    assert classify_emotion('I am so happy today!') == 'joy'
    assert classify_emotion('I am really scared of spiders.') == 'fear'
    assert classify_emotion('I am feeling a bit down today.') == 'sadness'
    assert classify_emotion('I am so angry at you!') == 'anger'
    assert classify_emotion('I am surprised to see you here.') == 'surprise'
    assert classify_emotion('I am feeling neutral.') == 'neutral'

test_classify_emotion()