from f00385_create_optimizer import *
def test_create_optimizer():
    optimizer = create_optimizer(learning_rate=2e-5, weight_decay_rate=0.01)
    assert isinstance(optimizer, tf.keras.optimizers.Optimizer)

    # Add more test cases here

    print('All test cases pass')

test_create_optimizer()
