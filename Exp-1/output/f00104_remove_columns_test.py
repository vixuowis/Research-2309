from f00104_remove_columns import *
def test_remove_columns():
    dataset = datasets.load_dataset('lj_speech', split='train')
    columns_to_remove = ['file', 'id', 'normalized_text']
    modified_dataset = remove_columns(dataset, columns_to_remove)
    assert modified_dataset.column_names == ['speech', 'text']
    assert modified_dataset[0]['speech'] == 'path/to/audio1.wav'
    assert modified_dataset[0]['text'] == 'hello world'
    assert modified_dataset[1]['speech'] == 'path/to/audio2.wav'
    assert modified_dataset[1]['text'] == 'how are you'
    assert modified_dataset[2]['speech'] == 'path/to/audio3.wav'
    assert modified_dataset[2]['text'] == 'goodbye'

test_remove_columns()
