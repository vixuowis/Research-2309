from f00542_get_image_processing_info import *
def test_get_image_processing_info():
	image_processor = ImageProcessor()
	model = PretrainedModel()
	info = get_image_processing_info(image_processor, model)

	assert "mean" in info
	assert "std" in info
	assert "resize_to" in info
	assert "num_frames_to_sample" in info
	assert "sample_rate" in info
	assert "fps" in info
	assert "clip_duration" in info

	print("All tests passed.")
