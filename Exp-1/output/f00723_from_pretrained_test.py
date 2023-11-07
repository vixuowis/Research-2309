from f00723_from_pretrained import *
def test_from_pretrained():
    config_path = 'custom-resnet/config.json'
    config = from_pretrained(config_path)

    assert isinstance(config, ResnetConfig)
    assert config.hidden_size == 768
    assert config.num_hidden_layers == 12
    assert config.num_attention_heads == 12

    print('All tests passed!')


test_from_pretrained()
