# Test function for translate_spanish_to_polish
# The function is tested with a Spanish sentence
# The expected output is the Polish translation of the input sentence
# The test function uses assert to compare the function output with the expected output

def test_translate_spanish_to_polish():
    # Spanish sentence
    spanish_text = 'Hola, ¿cómo estás?'
    # Expected output
    expected_output = 'Cześć, jak się masz?'
    # Function output
    function_output = translate_spanish_to_polish(spanish_text)
    # Compare function output with expected output
    assert function_output == expected_output, f'Expected: {expected_output}, but got: {function_output}'

# Run the test function
test_translate_spanish_to_polish()