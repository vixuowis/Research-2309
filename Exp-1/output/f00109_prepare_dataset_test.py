from f00109_prepare_dataset import *
def test_prepare_dataset():
    assert np.array_equal(prepare_dataset(lj_speech[0]), expected_output)
    assert np.array_equal(prepare_dataset(lj_speech[1]), expected_output)
    assert np.array_equal(prepare_dataset(lj_speech[2]), expected_output)
    assert np.array_equal(prepare_dataset(lj_speech[3]), expected_output)
    assert np.array_equal(prepare_dataset(lj_speech[4]), expected_output)

if __name__ == '__main__':
    test_prepare_dataset()
