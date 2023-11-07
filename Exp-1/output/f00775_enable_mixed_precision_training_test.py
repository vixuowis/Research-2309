from f00775_enable_mixed_precision_training import *
def test_enable_mixed_precision_training():
    training_args = TrainingArguments(per_device_train_batch_size=4)
    updated_training_args = enable_mixed_precision_training(training_args)
    assert updated_training_args.fp16 == True


test_enable_mixed_precision_training()
