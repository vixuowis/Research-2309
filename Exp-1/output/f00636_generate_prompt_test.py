from f00636_generate_prompt import *
def test_generate_prompt():
    assert generate_prompt("What is the color of the car?") == "Question: What is the color of the car? Answer:"
    assert generate_prompt("How many apples are there?") == "Question: How many apples are there? Answer:"
    assert generate_prompt("Is the sky blue?") == "Question: Is the sky blue? Answer:"
    assert generate_prompt("Where is the nearest supermarket?") == "Question: Where is the nearest supermarket? Answer:"
    assert generate_prompt("What time is it?") == "Question: What time is it? Answer:"

test_generate_prompt()
