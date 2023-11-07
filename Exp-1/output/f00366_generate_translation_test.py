from f00366_generate_translation import *
def test_generate_translation():
    inputs = "Hello, how are you?"
    expected_output = "Bonjour, comment ça va ?"
    assert generate_translation(inputs) == expected_output
    
    inputs = "What is your name?"
    expected_output = "Comment tu t'appelles ?"
    assert generate_translation(inputs) == expected_output
    
    inputs = "Where are you from?"
    expected_output = "D'où viens-tu ?"
    assert generate_translation(inputs) == expected_output
    
    inputs = "I love pizza."
    expected_output = "J'adore la pizza."
    assert generate_translation(inputs) == expected_output
    
    inputs = "Thank you very much!"
    expected_output = "Merci beaucoup !"
    assert generate_translation(inputs) == expected_output
    
    print("All test cases pass.")
