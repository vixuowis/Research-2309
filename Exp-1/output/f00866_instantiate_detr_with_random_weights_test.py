from f00866_instantiate_detr_with_random_weights import *
def test_instantiate_detr_with_random_weights():
    
    model = instantiate_detr_with_random_weights()
    
    assert isinstance(model, DetrForObjectDetection)
    print('Test Passed!')
    

test_instantiate_detr_with_random_weights()
