from f00843_create_dummy_dataset import *
def test_create_dummy_dataset():
    ds = create_dummy_dataset(512, 512)
    assert len(ds) == 512
    assert len(ds[0]['input_ids']) == 512
    assert ds[0]['labels'] in [0, 1]
    assert ds.format['type'] == 'torch'

test_create_dummy_dataset()
