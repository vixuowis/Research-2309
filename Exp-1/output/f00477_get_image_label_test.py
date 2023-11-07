from f00477_get_image_label import *
def test_get_image_label():
    data = {'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=512x512 at 0x7F52AFC8AC50>, 'label': 79}
    expected_result = (<PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=512x512 at 0x7F52AFC8AC50>, 79)
    assert get_image_label(data) == expected_result

test_get_image_label()
