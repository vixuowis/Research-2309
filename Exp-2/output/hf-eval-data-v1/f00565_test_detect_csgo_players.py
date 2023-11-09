def test_detect_csgo_players():
    """
    This function tests the 'detect_csgo_players' function by using a sample image and checking if the output is not None.
    """
    # URL of a sample image
    image = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
    
    # Call the 'detect_csgo_players' function
    render = detect_csgo_players(image)
    
    # Check if the output is not None
    assert render is not None, 'No players detected in the image.'
    
    print('Test passed.')

# Run the test function
test_detect_csgo_players()