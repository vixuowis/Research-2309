def test_extract_names_and_locations():
    # Test the function with a sample chat room conversation
    text = "Hello, my name is John and I live in New York."
    result = extract_names_and_locations(text)
    # Check if the function correctly extracted the name and location
    assert 'John' in result and 'New York' in result
    # Test the function with another sample chat room conversation
    text = "I am from London and my name is Emma."
    result = extract_names_and_locations(text)
    # Check if the function correctly extracted the name and location
    assert 'Emma' in result and 'London' in result

test_extract_names_and_locations()