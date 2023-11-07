from f00448_load_feature_extractor import *
def test_load_feature_extractor():
    assert load_feature_extractor("stevhliu/my_awesome_minds_model") is not None

test_load_feature_extractor()
