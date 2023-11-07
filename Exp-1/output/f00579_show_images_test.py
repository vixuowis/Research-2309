from f00579_show_images import *
def test_show_images():
	# Test case 1
	image_target = load_image('target.jpg')
	query_image = load_image('query.jpg')
	show_images(image_target, query_image)

	# Test case 2
	image_target = load_image('target2.jpg')
	query_image = load_image('query2.jpg')
	show_images(image_target, query_image)

	# Test case 3
	image_target = load_image('target3.jpg')
	query_image = load_image('query3.jpg')
	show_images(image_target, query_image)

	# Test case 4
	image_target = load_image('target4.jpg')
	query_image = load_image('query4.jpg')
	show_images(image_target, query_image)

	# Test case 5
	image_target = load_image('target5.jpg')
	query_image = load_image('query5.jpg')
	show_images(image_target, query_image)

if __name__ == '__main__':
	test_show_images()
