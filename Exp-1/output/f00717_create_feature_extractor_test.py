from f00717_create_feature_extractor import *
def test_create_feature_extractor():
    feature_extractor = create_feature_extractor()
    assert isinstance(feature_extractor, Wav2Vec2FeatureExtractor)
    assert feature_extractor.do_normalize == True
    assert feature_extractor.feature_extractor_type == 'Wav2Vec2FeatureExtractor'
    assert feature_extractor.feature_size == 1
    assert feature_extractor.padding_side == 'right'
    assert feature_extractor.padding_value == 0.0
    assert feature_extractor.return_attention_mask == False
    assert feature_extractor.sampling_rate == 16000
