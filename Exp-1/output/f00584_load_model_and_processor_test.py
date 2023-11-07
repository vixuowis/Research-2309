from f00584_load_model_and_processor import *
def test_load_model_and_processor():
    checkpoint = "checkpoint_name"
    model, processor = load_model_and_processor(checkpoint)
    assert isinstance(model, AutoModelForZeroShotImageClassification)
    assert isinstance(processor, AutoProcessor)


test_load_model_and_processor()
