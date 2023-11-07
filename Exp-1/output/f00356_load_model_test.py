from f00356_load_model import *
def test_load_model():
	# Test Case 1
	checkpoint = 'path/to/checkpoint'
	model = load_model(checkpoint)
	assert isinstance(model, TFAutoModelForSeq2SeqLM)

	# Test Case 2
	checkpoint = 'path/to/another/checkpoint'
	model = load_model(checkpoint)
	assert isinstance(model, TFAutoModelForSeq2SeqLM)

	# Test Case 3
	checkpoint = 'path/to/yet/another/checkpoint'
	model = load_model(checkpoint)
	assert isinstance(model, TFAutoModelForSeq2SeqLM)

	# Test Case 4
	checkpoint = 'path/to/last/checkpoint'
	model = load_model(checkpoint)
	assert isinstance(model, TFAutoModelForSeq2SeqLM)

	# Test Case 5
	checkpoint = 'path/to/final/checkpoint'
	model = load_model(checkpoint)
	assert isinstance(model, TFAutoModelForSeq2SeqLM)


test_load_model()
