from f00517_transforms import *
import tensorflow as tf


def test_transforms():
    # Test case 1
    image1 = tf.constant([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    transformed_image1 = transforms(image1)
    assert transformed_image1.shape == (3, 1, 3)

    # Test case 2
    image2 = tf.constant([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]]])
    transformed_image2 = transforms(image2)
    assert transformed_image2.shape == (3, 2, 3)

    # Test case 3
    image3 = tf.constant([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24], [25, 26, 27]]])
    transformed_image3 = transforms(image3)
    assert transformed_image3.shape == (3, 3, 3)


if __name__ == '__main__':
    test_transforms()
