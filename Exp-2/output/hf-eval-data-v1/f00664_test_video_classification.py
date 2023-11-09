def test_video_classification():
    """
    This function tests the video_classification function by using a sample video.
    """
    # Define the path to the sample video
    video_path = 'path/to/sample/video'
    
    # Call the video_classification function
    predicted_class = video_classification(video_path)
    
    # Assert that the function returns a string (the predicted class)
    assert isinstance(predicted_class, str), 'The function should return a string.'
    
    # Print the result
    print(f'The predicted class of the video content is: {predicted_class}')

# Call the test function
test_video_classification()