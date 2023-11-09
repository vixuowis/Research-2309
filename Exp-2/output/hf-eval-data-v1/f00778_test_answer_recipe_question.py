def test_answer_recipe_question():
    """
    This function tests the 'answer_recipe_question' function.
    It uses a sample question and image, and checks if the returned answer is a string.
    """
    question_text = 'What is the main ingredient in this recipe?'
    recipe_image = 'sample_image.jpg'  # replace with a valid image file
    answer = answer_recipe_question(question_text, recipe_image)
    assert isinstance(answer, str), 'The answer should be a string.'

test_answer_recipe_question()