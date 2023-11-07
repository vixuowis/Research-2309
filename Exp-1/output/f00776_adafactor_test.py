from f00776_adafactor import *
training_args = TrainingArguments(per_device_train_batch_size=4, optim="adafactor", **default_args)
