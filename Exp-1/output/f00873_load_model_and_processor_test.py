from f00873_load_model_and_processor import *
def test_load_model_and_processor():
	model_id = "facebook/mms-1b-all"
	processor, model = load_model_and_processor(model_id)
	assert isinstance(processor, AutoProcessor)
	assert isinstance(model, Wav2Vec2ForCTC)

	# Additional test cases
	# ...

if __name__ == '__main__':
	test_load_model_and_processor()
