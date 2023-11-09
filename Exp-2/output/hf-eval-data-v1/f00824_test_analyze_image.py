def test_analyze_image():
    """
    Test the analyze_image function.
    """
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    last_hidden_states = analyze_image(url)
    assert last_hidden_states is not None, 'No output from the model'
    assert last_hidden_states.size(0) == 1, 'Unexpected output dimensions from the model'
test_analyze_image()