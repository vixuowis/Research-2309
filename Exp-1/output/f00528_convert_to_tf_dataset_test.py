from f00528_convert_to_tf_dataset import *
def test_convert_to_tf_dataset():
    train_ds = ...
    test_ds = ...
    columns = ['pixel_values', 'label']
    shuffle = True
    batch_size = 32

    tf_train_dataset = convert_to_tf_dataset(train_ds, columns, shuffle, batch_size)
    tf_eval_dataset = convert_to_tf_dataset(test_ds, columns, shuffle, batch_size)

    assert tf_train_dataset is not None
    assert tf_eval_dataset is not None


test_convert_to_tf_dataset()
