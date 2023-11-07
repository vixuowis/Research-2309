from f00748_translate_question import *
def test_translate_question():
    assert translate_question("Le chat est noir", "French", "English") == "The cat is black"
    assert translate_question("Comment ça va ?", "French", "English") == "How are you?"
    assert translate_question("Quelle est la capitale de la France ?", "French", "English") == "What is the capital of France?"
    assert translate_question("Combien ça coûte ?", "French", "English") == "How much does it cost?"
    assert translate_question("Où est la bibliothèque ?", "French", "English") == "Where is the library?"


test_translate_question()
