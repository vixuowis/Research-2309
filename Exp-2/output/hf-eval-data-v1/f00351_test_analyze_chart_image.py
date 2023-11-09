def test_analyze_chart_image():
    '''
    This function tests the analyze_chart_image function.
    It uses a sample chart image URL and checks if the output is a string.
    '''
    # Sample chart image URL
    url = "https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/val/png/5090.png"

    # Call the analyze_chart_image function
    summary = analyze_chart_image(url)

    # Check if the output is a string
    assert isinstance(summary, str), "The function should return a string."

    # Check if the output is not empty
    assert len(summary) > 0, "The function should return a non-empty string."

test_analyze_chart_image()