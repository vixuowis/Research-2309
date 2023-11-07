from f00582_classify_image import *
def test_classify_image():
	url = "https://unsplash.com/photos/g8oS8-82DxI/download?ixid=MnwxMjA3fDB8MXx0b3BpY3x8SnBnNktpZGwtSGt8fHx8fDJ8fDE2NzgxMDYwODc&force=true&w=640"
	image = classify_image(url)
	assert isinstance(image, Image.Image)

	# Add more test cases here
