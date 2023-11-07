from f00601_remove_ocr_features import *
def test_remove_ocr_features():
    dataset = Dataset.from_dict({...})
    updated_dataset = remove_ocr_features(dataset)
    assert 'words' not in updated_dataset.column_names
    assert 'bounding_boxes' not in updated_dataset.column_names

test_remove_ocr_features()
