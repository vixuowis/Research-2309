from f00107_from_pretrained import *
import pytest

from transformers import AutoProcessor


@pytest.fixture(scope='session')
def processor():
    return AutoProcessor.from_pretrained('facebook/wav2vec2-base-960h')


def test_from_pretrained(processor):
    assert isinstance(processor, AutoProcessor)

