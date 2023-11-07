from f00369_generate_translation import *
def test_generate_translation():
    inputs = "Hello, how are you?"
    model_path = "my_awesome_opus_books_model"
    translation = generate_translation(inputs, model_path)
    assert isinstance(translation, str)
    assert len(translation) > 0

    inputs = "Goodbye, see you later."
    translation = generate_translation(inputs, model_path, do_sample=False)
    assert isinstance(translation, str)
    assert len(translation) > 0

    inputs = "Where is the nearest restaurant?"
    translation = generate_translation(inputs, model_path, top_k=10)
    assert isinstance(translation, str)
    assert len(translation) > 0

    inputs = "Can you help me with my homework?"
    translation = generate_translation(inputs, model_path, top_p=0.8)
    assert isinstance(translation, str)
    assert len(translation) > 0

    inputs = "I don't understand."
    translation = generate_translation(inputs, model_path, max_new_tokens=20)
    assert isinstance(translation, str)
    assert len(translation) > 0

