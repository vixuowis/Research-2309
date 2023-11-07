from f00357_prepare_tf_dataset import *
def test_prepare_tf_dataset():
    tf_train_set = model.prepare_tf_dataset(
        tokenized_books["train"],
        shuffle=True,
        batch_size=16,
        collate_fn=data_collator,
    )

    tf_test_set = model.prepare_tf_dataset(
        tokenized_books["test"],
        shuffle=False,
        batch_size=16,
        collate_fn=data_collator,
    )

    assert len(tf_train_set) == 16
    assert len(tf_test_set) == 16
    assert tf_train_set.shuffle == True
    assert tf_test_set.shuffle == False


test_prepare_tf_dataset()
