from f00089_feature_extractor import *
def test_feature_extractor():
    audio_input = [dataset[0]['audio']['array']]
    expected_output = {'input_values': [np.array([3.8106556e-04, 2.7506407e-03, 2.8015103e-03, ..., 5.6335266e-04, 4.6588284e-06, -1.7142107e-04], dtype=np.float32)]}
    assert feature_extractor(audio_input, sampling_rate=16000) == expected_output

test_feature_extractor()
