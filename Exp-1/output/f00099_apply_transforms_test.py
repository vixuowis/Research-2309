from f00099_apply_transforms import *
def test_apply_transforms():
    dataset = Dataset.from_dict({
        'text': ['Hello', 'World'],
        'label': [0, 1]
    })
    transforms = {
        'text': [{'type': 'lowercase'}, {'type': 'remove_special_characters'}],
        'label': {'type': 'one_hot_encoding', 'num_classes': 2}
    }
    apply_transforms(dataset, transforms)
    assert dataset[0]['text'] == 'hello'
    assert dataset[1]['text'] == 'world'
    assert dataset[0]['label'] == [1, 0]
    assert dataset[1]['label'] == [0, 1]

test_apply_transforms()
