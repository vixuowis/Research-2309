def test_detect_gibberish():
    # Test the function with some sample text
    assert detect_gibberish('I love AutoNLP') is not None
    assert detect_gibberish('asdklfj asldkfj asldkfj') is not None
    # Load the dataset used for performance testing
    dataset = load_dataset('madhurjindal/autonlp-data-Gibberish-Detector')
    # Select a few samples from the dataset
    samples = dataset['train'][:5]
    # Test the function with the samples
    for sample in samples:
        assert detect_gibberish(sample['text']) is not None