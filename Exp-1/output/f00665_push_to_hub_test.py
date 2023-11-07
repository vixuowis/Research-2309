from f00665_push_to_hub import *
def test_push_to_hub(self):
    model_url = "https://huggingface.co/models/username/model_name"
    with patch.object(self.model, "push_to_hub", return_value=model_url):
        result = self.trainer.push_to_hub()
    assert result == model_url

