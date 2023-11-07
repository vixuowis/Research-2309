from f00836_generate_text import *
def test_generate_text():
    prompt = 'Hugging Face is a community-based open-source platform for machine learning.'
    generated_text = generate_text(prompt)
    assert isinstance(generated_text, str)

    prompt = 'This is a test.'
    generated_text = generate_text(prompt)
    assert isinstance(generated_text, str)

    prompt = 'Hello, world!'
    generated_text = generate_text(prompt)
    assert isinstance(generated_text, str)

    prompt = 'Lorem ipsum dolor sit amet.'
    generated_text = generate_text(prompt)
    assert isinstance(generated_text, str)

    prompt = '12345'
    generated_text = generate_text(prompt)
    assert isinstance(generated_text, str)
