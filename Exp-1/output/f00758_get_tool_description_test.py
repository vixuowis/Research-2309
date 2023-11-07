from f00758_get_tool_description import *
def test_get_tool_description():
    assert get_tool_description('controlnet_transformer') == (controlnet_transformer.description, controlnet_transformer.name)
    assert get_tool_description('non_existing_tool') == ('', '')
    assert get_tool_description('') == ('', '')


test_get_tool_description()
