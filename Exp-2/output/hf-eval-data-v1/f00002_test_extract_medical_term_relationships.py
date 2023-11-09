def test_extract_medical_term_relationships():
    """
    This function tests the 'extract_medical_term_relationships' function.
    It uses a sample medical term and checks if the output is a tensor.
    """
    sample_term = 'diabetes'
    output = extract_medical_term_relationships(sample_term)

    assert isinstance(output, torch.Tensor), 'Output should be a tensor.'

    print('All tests passed.')

test_extract_medical_term_relationships()