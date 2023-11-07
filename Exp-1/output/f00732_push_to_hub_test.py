from f00732_push_to_hub import *
def test_push_to_hub():
    model = CustomResNet50D()
    repo_id = "custom-resnet50d"
    result = model.push_to_hub(repo_id)
    assert result.startswith("https://huggingface.co/")


test_push_to_hub()

