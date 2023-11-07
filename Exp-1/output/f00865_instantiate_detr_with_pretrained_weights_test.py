from f00865_instantiate_detr_with_pretrained_weights import *
def test_instantiate_detr_with_pretrained_weights():
    config = DetrConfig()
    model = DetrForObjectDetection(config)
    assert isinstance(config, DetrConfig)
    assert isinstance(model, DetrForObjectDetection)
