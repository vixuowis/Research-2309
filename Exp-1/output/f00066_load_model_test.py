from f00066_load_model import *
import bitsandbytes
import assert

def test_load_model():
    model_path = "path/to/model"
    model = load_model(model_path)
    assert isinstance(model, bitsandbytes.Model)

    model_8bit = load_model(model_path, load_in_8bit=True)
    assert isinstance(model_8bit, bitsandbytes.Model)
