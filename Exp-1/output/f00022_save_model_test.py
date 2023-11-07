from f00022_save_model import *
def test_save_model():
	assert save_model(model, tokenizer, save_directory) == None


def test_all():
	test_save_model()

test_all()
