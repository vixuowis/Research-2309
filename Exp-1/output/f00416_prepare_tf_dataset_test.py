from f00416_prepare_tf_dataset import *
def test_prepare_tf_dataset():
    data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)
    tf_train_set = model.prepare_tf_dataset(
        tokenized_swag["train"],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator,
    )

    tf_validation_set = model.prepare_tf_dataset(
        tokenized_swag["validation"],
        shuffle=False,
        batch_size=batch_size,
        collate_fn=data_collator,
    )

    assert isinstance(tf_train_set, tf.data.Dataset)
    assert isinstance(tf_validation_set, tf.data.Dataset)

    # Add more assertions as needed

    print("All tests passed!")

test_prepare_tf_dataset()
