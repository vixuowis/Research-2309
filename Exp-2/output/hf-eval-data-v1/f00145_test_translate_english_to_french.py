def test_translate_english_to_french():
    '''
    Tests the function translate_english_to_french.
    '''
    # Define the test case
    english_text = 'Welcome to our hotel, we hope you enjoy your stay.'
    expected_french_text = 'Bienvenue dans notre hôtel, nous espérons que vous apprécierez votre séjour.'
    
    # Call the function with the test case
    french_text = translate_english_to_french(english_text)
    
    # Assert that the function's output is as expected
    assert french_text == expected_french_text, f'Expected: {expected_french_text}, but got: {french_text}'

# Run the test function
test_translate_english_to_french()