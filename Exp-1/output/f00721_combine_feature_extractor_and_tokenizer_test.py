from f00721_combine_feature_extractor_and_tokenizer import *
def test_combine_feature_extractor_and_tokenizer():
    feature_extractor = FeatureExtractor()
    tokenizer = Tokenizer()
    processor = combine_feature_extractor_and_tokenizer(feature_extractor, tokenizer)
    assert isinstance(processor, Wav2Vec2Processor)
    assert processor.feature_extractor == feature_extractor
    assert processor.tokenizer == tokenizer

if __name__ == '__main__':
    test_combine_feature_extractor_and_tokenizer()
