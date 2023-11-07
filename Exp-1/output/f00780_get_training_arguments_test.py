from f00780_get_training_arguments import *
def test_get_training_arguments():
    training_args = get_training_arguments()
    assert training_args.per_device_train_batch_size == 1
    assert training_args.gradient_accumulation_steps == 4
    assert training_args.gradient_checkpointing
    assert training_args.fp16
    # Add more test cases if needed
