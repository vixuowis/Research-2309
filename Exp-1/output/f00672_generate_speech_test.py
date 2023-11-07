from f00672_generate_speech import *
import numpy as np
import assert

input_ids = [1, 2, 3]
speaker_embeddings = [0.1, 0.2, 0.3]

result = generate_speech(input_ids, speaker_embeddings)
expected_result = np.array([0.4, 0.5, 0.6])

assert np.array_equal(result, expected_result)

input_ids = [4, 5, 6]
speaker_embeddings = [0.4, 0.5, 0.6]

result = generate_speech(input_ids, speaker_embeddings)
expected_result = np.array([0.7, 0.8, 0.9])

assert np.array_equal(result, expected_result)

input_ids = [7, 8, 9]
speaker_embeddings = [0.7, 0.8, 0.9]

result = generate_speech(input_ids, speaker_embeddings)
expected_result = np.array([1.0, 1.1, 1.2])

assert np.array_equal(result, expected_result)
