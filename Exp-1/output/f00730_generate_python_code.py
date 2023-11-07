from typing import *
def generate_python_code():
    resnet50d_config = ResnetConfig(block_type="bottleneck", stem_width=32, stem_type="deep", avg_down=True)
    resnet50d = ResnetModelForImageClassification(resnet50d_config)

    pretrained_model = timm.create_model("resnet50d", pretrained=True)
    resnet50d.model.load_state_dict(pretrained_model.state_dict())

    python_code = """
    resnet50d_config = ResnetConfig(block_type=\"bottleneck\", stem_width=32, stem_type=\"deep\", avg_down=True)
    resnet50d = ResnetModelForImageClassification(resnet50d_config)

    pretrained_model = timm.create_model(\"resnet50d\", pretrained=True)
    resnet50d.model.load_state_dict(pretrained_model.state_dict())
    """

    return python_code
