from f00883_docvqa import *
def test_docvqa():
    question = "When is the coffee break?"
    image_path = "path/to/image.jpg"

    result = docvqa(question, image_path)
    assert result == {'question': 'When is the coffee break?', 'answer': '11-14 to 11:39 a.m.'}


test_docvqa()
