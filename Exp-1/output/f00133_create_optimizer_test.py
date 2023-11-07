from f00133_create_optimizer import *
def test_create_optimizer():
    model = YourModel()
    learning_rate = 5e-5
    optimizer = create_optimizer(model, learning_rate)
    assert isinstance(optimizer, AdamW)
    assert optimizer.param_groups[0]['lr'] == learning_rate


test_create_optimizer()
