from f00774_enable_gradient_checkpointing import *
def test_enable_gradient_checkpointing():
    training_args = TrainingArguments(per_device_train_batch_size=1, gradient_accumulation_steps=4)
    modified_args = enable_gradient_checkpointing(training_args)
    assert modified_args.gradient_checkpointing == True

test_enable_gradient_checkpointing()
