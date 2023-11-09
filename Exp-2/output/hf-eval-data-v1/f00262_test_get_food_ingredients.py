def test_get_food_ingredients():
    '''
    This function tests the get_food_ingredients function by using a sample image URL and question.
    '''
    img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
    question = 'What are the ingredients of this dish?'
    
    ingredient_info = get_food_ingredients(img_url, question)
    
    assert isinstance(ingredient_info, str), 'The output should be a string.'
    assert len(ingredient_info) > 0, 'The output string should not be empty.'

test_get_food_ingredients()