from f00026_convert_model_from_tf import *
def test_convert_model_from_tf():
	# Test case 1
	tf_save_directory = "/path/to/tf_model"
	pt_model = convert_model_from_tf(tf_save_directory)
	assert isinstance(pt_model, AutoModelForSequenceClassification)

	# Test case 2
	tf_save_directory = "/path/to/another_tf_model"
	pt_model = convert_model_from_tf(tf_save_directory)
	assert isinstance(pt_model, AutoModelForSequenceClassification)

	# Test case 3
	tf_save_directory = "/path/to/yet_another_tf_model"
	pt_model = convert_model_from_tf(tf_save_directory)
	assert isinstance(pt_model, AutoModelForSequenceClassification)

	# Test case 4
	tf_save_directory = "/path/to/one_more_tf_model"
	pt_model = convert_model_from_tf(tf_save_directory)
	assert isinstance(pt_model, AutoModelForSequenceClassification)

	# Test case 5
	tf_save_directory = "/path/to/final_tf_model"
	pt_model = convert_model_from_tf(tf_save_directory)
	assert isinstance(pt_model, AutoModelForSequenceClassification)


test_convert_model_from_tf()
