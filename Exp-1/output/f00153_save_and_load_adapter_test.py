from f00153_save_and_load_adapter import *
def test_save_and_load_adapter():
    save_dir = 'path/to/save/dir'
    model = AutoModelForCausalLM()
    save_and_load_adapter(save_dir)
    loaded_model = AutoModelForCausalLM.from_pretrained(save_dir)
    assert model.config == loaded_model.config
    assert model.state_dict().keys() == loaded_model.state_dict().keys()
    assert model.adapter == loaded_model.adapter

test_save_and_load_adapter()
