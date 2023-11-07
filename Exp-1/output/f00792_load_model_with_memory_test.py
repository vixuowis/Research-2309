from f00792_load_model_with_memory import *
def test_load_model_with_memory():
    max_memory_mapping = {0: "600MB", 1: "1GB"}
    model_name = "bigscience/bloom-3b"
    device_map = "auto"
    load_in_4bit = True
    model = load_model_with_memory(model_name, device_map, load_in_4bit, max_memory_mapping)
    assert isinstance(model, AutoModelForCausalLM)


test_load_model_with_memory()
