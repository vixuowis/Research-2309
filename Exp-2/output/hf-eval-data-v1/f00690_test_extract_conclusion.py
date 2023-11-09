def test_extract_conclusion():
    # Test the extract_conclusion function
    text = 'Studies have been shown that owning a dog is good for you. Having a dog can help decrease stress levels, improve your mood, and increase physical activity.'
    conclusion = extract_conclusion(text)
    print(conclusion)
    # Assert that the conclusion is not empty
    assert conclusion != ''

    # Test with another text
    text = 'Regular exercise can prevent and reverse age-related decreases in muscle mass and strength, improve balance, flexibility, and endurance, and decrease the risk of falls in the elderly.'
    conclusion = extract_conclusion(text)
    print(conclusion)
    # Assert that the conclusion is not empty
    assert conclusion != ''

test_extract_conclusion()