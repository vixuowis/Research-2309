from f00759_instantiate_agent import *
def test_instantiate_agent():
    controlnet_transformer = "controlnet_transformer"
    upscaler = "upscaler"
    agent = instantiate_agent(controlnet_transformer, upscaler)
    assert isinstance(agent, HfAgent)


test_instantiate_agent()
