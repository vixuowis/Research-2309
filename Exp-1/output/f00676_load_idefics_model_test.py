from f00676_load_idefics_model import *
def test_load_idefics_model():
    checkpoint = "/path/to/checkpoint"
    processor, model = load_idefics_model(checkpoint)

    # Test processor
    assert isinstance(processor, AutoProcessor)

    # Test model
    assert isinstance(model, IdeficsForVisionText2Text)

    print("All tests pass.")
