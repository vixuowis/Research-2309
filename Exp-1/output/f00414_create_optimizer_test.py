from f00414_create_optimizer import *
def test_create_optimizer():
    init_lr = 5e-5
    num_warmup_steps = 0
    num_train_steps = 1000
    optimizer, schedule = create_optimizer(init_lr, num_warmup_steps, num_train_steps)
    assert isinstance(optimizer, AdamWeightDecay)
    assert isinstance(schedule, tf.keras.optimizers.schedules.LearningRateSchedule)

test_create_optimizer()
