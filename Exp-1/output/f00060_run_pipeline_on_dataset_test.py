from f00060_run_pipeline_on_dataset import *
def test_run_pipeline_on_dataset():
    data = ['My example 0', 'My example 1', 'My example 2', 'My example 3', 'My example 4']
    model = 'gpt2'
    device = 0
    expected_result = 75
    assert run_pipeline_on_dataset(data, model, device) == expected_result

test_run_pipeline_on_dataset()
