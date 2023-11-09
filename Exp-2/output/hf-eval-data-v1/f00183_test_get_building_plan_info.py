def test_get_building_plan_info():
    """
    This function tests the 'get_building_plan_info' function.
    """
    # Define a sample question and building plan data
    question = 'What is the total estimated cost of the project?'
    building_plan_data = 'The total estimated cost of the project is $1,000,000.'

    # Call the function with the sample data
    answer = get_building_plan_info(question, building_plan_data)

    # Assert that the function returns the correct answer
    assert '1,000,000' in answer, f'Error: {answer}'

# Run the test function
test_get_building_plan_info()