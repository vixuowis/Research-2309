def test_detect_csgo_players():
    """
    Tests the detect_csgo_players function by using a sample game image.
    """
    game_image = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
    result = detect_csgo_players(game_image)
    assert result is not None, 'No result from the function.'
    assert isinstance(result, type(render_result)), 'The result is not a rendered image.'

test_detect_csgo_players()