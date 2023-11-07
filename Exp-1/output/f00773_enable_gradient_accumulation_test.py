from f00773_enable_gradient_accumulation import *
def test_enable_gradient_accumulation():
    training_args = TrainingArguments(per_device_train_batch_size=1, **default_args)
    accumulation_steps = 4
    expected_output = TrainingArguments(per_device_train_batch_size=1, gradient_accumulation_steps=4, **default_args)
    assert enable_gradient_accumulation(training_args, accumulation_steps) == expected_output


test_enable_gradient_accumulation()
