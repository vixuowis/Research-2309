def test_generate_plant_care_instruction():
    # Define a prompt for the test
    prompt = 'I have a potted plant and I want to take care of it. What should I do?'
    # Call the function with the test prompt
    instructions = generate_plant_care_instruction(prompt)
    # Check if the function returns a string
    assert isinstance(instructions, str), 'The function should return a string.'
    # Check if the function returns a non-empty string
    assert len(instructions) > 0, 'The function should return a non-empty string.'

test_generate_plant_care_instruction()