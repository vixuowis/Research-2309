from f00140_backward import *
def test_backward():
    accelerator = Accelerator()
    loss = 0.5
    backward(loss)
    assert accelerator.backward_called == True

test_backward()
