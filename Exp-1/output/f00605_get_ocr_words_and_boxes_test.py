from f00605_get_ocr_words_and_boxes import *
def test_get_ocr_words_and_boxes():
    example1 = {'image_path': 'image1.jpg'}
    example2 = {'image_path': 'image2.jpg'}
    example3 = {'image_path': 'image3.jpg'}
    expected1 = {'image_path': 'image1.jpg', 'ocr_words': ['word1', 'word2', 'word3'], 'ocr_boxes': 'box1 box2 box3'}
    expected2 = {'image_path': 'image2.jpg', 'ocr_words': ['word4', 'word5', 'word6'], 'ocr_boxes': 'box4 box5 box6'}
    expected3 = {'image_path': 'image3.jpg', 'ocr_words': ['word7', 'word8', 'word9'], 'ocr_boxes': 'box7 box8 box9'}
    assert get_ocr_words_and_boxes(example1) == expected1
    assert get_ocr_words_and_boxes(example2) == expected2
    assert get_ocr_words_and_boxes(example3) == expected3


def main():
    test_get_ocr_words_and_boxes()


if __name__ == '__main__':
    main()
