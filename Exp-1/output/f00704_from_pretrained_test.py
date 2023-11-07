from f00704_from_pretrained import *
def test_from_pretrained():
    model_name_or_path = "distilbert-base-uncased"
    config = my_config
    cache_dir = None
    model_args = ()
    kwargs = {}
    model = from_pretrained(model_name_or_path, config, cache_dir, *model_args, **kwargs)
    assert isinstance(model, PreTrainedModel)

test_from_pretrained()
