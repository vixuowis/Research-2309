from f00122_load_and_compile_model import *
def test_load_and_compile_model():
    model_name = 'bert-base-uncased'
    num_labels = 2
    
    model = load_and_compile_model(model_name, num_labels)
    
    assert isinstance(model, TFAutoModelForSequenceClassification)
    assert model.num_labels == num_labels
    assert model.optimizer.__class__.__name__ == 'Adam'
    assert model.loss.__class__.__name__ == 'CrossEntropyLoss'
    assert 'accuracy' in model.metrics
    
    print('All tests passed!')
    

test_load_and_compile_model()
