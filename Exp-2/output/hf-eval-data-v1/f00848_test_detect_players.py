def test_detect_players():
    """
    Test the detect_players function.
    """
    image_path = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
    detected_players = detect_players(image_path)
    assert isinstance(detected_players, list), 'The result should be a list.'
    assert len(detected_players) > 0, 'There should be at least one player detected.'

test_detect_players()