from f00657_prepare_dataset import *
import torchaudio


def test_prepare_dataset():
    example = {'file': 'audio.wav'}
    output = prepare_dataset(example)
    assert 'file' in output
    assert 'spectrogram' in output
    assert isinstance(output['spectrogram'], torch.Tensor)


def test_prepare_dataset_multiple_examples():
    examples = [{'file': 'audio1.wav'}, {'file': 'audio2.wav'}, {'file': 'audio3.wav'}]
    outputs = [prepare_dataset(example) for example in examples]
    assert len(outputs) == len(examples)
    for output in outputs:
        assert 'file' in output
        assert 'spectrogram' in output
        assert isinstance(output['spectrogram'], torch.Tensor)


test_prepare_dataset()
test_prepare_dataset_multiple_examples()
