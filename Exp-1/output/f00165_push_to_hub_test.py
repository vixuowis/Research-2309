from f00165_push_to_hub import *
def test_push_to_hub(self):
    model = MyModel()
    repo_name = "my-awesome-model"
    url = model.push_to_hub(repo_name)
    assert url == "https://huggingface.co/my-awesome-model"
