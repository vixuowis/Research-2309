def test_generate_output():
    input_ids = torch.tensor([[7592]])
    output = generate_output(input_ids, model)
    expected_output = torch.tensor([[-0.1008, -0.4061]])
    assert torch.allclose(output, expected_output)

test_generate_output()
