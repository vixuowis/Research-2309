from f00511_get_image_info import *
def test_get_image_info():
    image_dict = {'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=512x683 at 0x7F9B0C201F90>, 'annotation': <PIL.PngImagePlugin.PngImageFile image mode=L size=512x683 at 0x7F9B0C201DD0>, 'scene_category': 368}
    expected_output = {'image': '<PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=512x683 at 0x7F9B0C201F90>', 'mode': 'RGB', 'size': (512, 683), 'location': '0x7F9B0C201F90'}
    assert get_image_info(image_dict) == expected_output


test_get_image_info()
