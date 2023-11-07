from f00417_compile import *
def test_compile():
    model = Model()
    optimizer = 'adam'
    model.compile(optimizer=optimizer)

    # Test case 1
    assert model.optimizer == optimizer

    # Test case 2
    assert model.loss is None

    # Test case 3
    assert model.metrics == []

test_compile()
