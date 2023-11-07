from f00088_load_feature_extractor import *
def test_load_feature_extractor():
    feature_extractor = load_feature_extractor("facebook/wav2vec2-base")
    assert isinstance(feature_extractor, AutoFeatureExtractor)

test_load_feature_extractor()
