from f00045_load_local_t5_model import *
def test_load_local_t5_model():
    model_path = "./path/to/local/directory"
    model = load_local_t5_model(model_path)
    
    # Test case 1
    assert isinstance(model, T5Model)
    
    # Test case 2
    assert model.config.architectures == ['T5Model']
    
    # Test case 3
    assert model.config.model_type == 't5'
    
    # Test case 4
    assert model.config.num_layers == 12
    
    # Test case 5
    assert model.config.d_model == 768
    
    print("All test cases pass")
