from f00402_load_swag_dataset import *
def test_load_swag_dataset():
    # Test 1
    swag = load_swag_dataset()
    assert len(swag) == 73546
    assert swag[0] == {'context': 'A person on a horse jumps over a broken down airplane.', 'endings': ['is scared of the possi...}
