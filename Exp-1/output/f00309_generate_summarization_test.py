from f00309_generate_summarization import *
def test_generate_summarization():
    model = TFAutoModelForCausalLM.from_pretrained("my_awesome_eli5_clm-model")
    input_ids = [1, 2, 3, 4, 5]
    outputs = generate_summarization(input_ids)
    assert len(outputs) > 0
    assert isinstance(outputs, tf.Tensor)
    print("All tests pass.")


test_generate_summarization()
