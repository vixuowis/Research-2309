from f00790_run_fp4_model_single_gpu import *
def test_run_fp4_model_single_gpu():
    model_name = 'bigscience/bloom-2b5'
    model = run_fp4_model_single_gpu(model_name)

    assert isinstance(model, AutoModelForCausalLM)
    assert model.device.type == 'cuda'
    assert model.device.index == 0

    print('All tests pass!')

test_run_fp4_model_single_gpu()
