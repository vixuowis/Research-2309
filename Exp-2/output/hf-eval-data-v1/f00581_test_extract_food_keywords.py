# Test function for extract_food_keywords
# Uses a sample input to test the function
# Asserts that the function returns a list

def test_extract_food_keywords():
    sample_input = "Today's meal: Fresh olive poke bowl topped with chia seeds. Very delicious!"
    result = extract_food_keywords(sample_input)
    assert isinstance(result, list), 'Function should return a list'
    assert len(result) > 0, 'Function should return at least one entity'

test_extract_food_keywords()