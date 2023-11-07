from f00147_set_adapter import *
def test_set_adapter():
    set_adapter('adapter_name')
    assert PeftModel.adapter == 'adapter_name'

test_set_adapter()
