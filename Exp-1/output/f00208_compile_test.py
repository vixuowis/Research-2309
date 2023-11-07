from f00208_compile import *
def test_compile():
    # Test case 1
    optimizer = tf.keras.optimizers.Adam()
    compile(optimizer)

    # Test case 2
    optimizer = tf.keras.optimizers.SGD()
    compile(optimizer)

    # Test case 3
    optimizer = tf.keras.optimizers.RMSprop()
    compile(optimizer)

    # Test case 4
    optimizer = tf.keras.optimizers.Adagrad()
    compile(optimizer)

    # Test case 5
    optimizer = tf.keras.optimizers.Adamax()
    compile(optimizer)

test_compile()
