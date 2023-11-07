from f00555_formatted_anns import *
def test_formatted_anns():
    # Test Case 1
    image_id = 1
    category = [1, 2, 3]
    area = [10.5, 20.3, 15.7]
    bbox = [[0, 0, 10, 10], [5, 5, 15, 15], [10, 10, 20, 20]]
    expected_result = [{'image_id': 1, 'category_id': 1, 'isCrowd': 0, 'area': 10.5, 'bbox': [0, 0, 10, 10]}, {'image_id': 1, 'category_id': 2, 'isCrowd': 0, 'area': 20.3, 'bbox': [5, 5, 15, 15]}, {'image_id': 1, 'category_id': 3, 'isCrowd': 0, 'area': 15.7, 'bbox': [10, 10, 20, 20]}]
    assert formatted_anns(image_id, category, area, bbox) == expected_result

    # Test Case 2
    image_id = 2
    category = [4, 5]
    area = [12.1, 18.9]
    bbox = [[2, 2, 12, 12], [8, 8, 18, 18]]
    expected_result = [{'image_id': 2, 'category_id': 4, 'isCrowd': 0, 'area': 12.1, 'bbox': [2, 2, 12, 12]}, {'image_id': 2, 'category_id': 5, 'isCrowd': 0, 'area': 18.9, 'bbox': [8, 8, 18, 18]}]
    assert formatted_anns(image_id, category, area, bbox) == expected_result

    # Test Case 3
    image_id = 3
    category = [6]
    area = [7.8]
    bbox = [[3, 3, 13, 13]]
    expected_result = [{'image_id': 3, 'category_id': 6, 'isCrowd': 0, 'area': 7.8, 'bbox': [3, 3, 13, 13]}]
    assert formatted_anns(image_id, category, area, bbox) == expected_result

test_formatted_anns()
