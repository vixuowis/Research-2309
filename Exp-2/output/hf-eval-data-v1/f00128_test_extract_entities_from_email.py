def test_extract_entities_from_email():
    '''
    This function tests the extract_entities_from_email function by using a sample sentence.
    '''
    sample_text = 'On September 1st George Washington won 1 dollar.'
    entities = extract_entities_from_email(sample_text)
    
    assert len(entities) > 0, 'No entities were extracted.'
    assert any('George Washington' in str(entity) for entity in entities), 'Expected entity not found.'
    assert any('1 dollar' in str(entity) for entity in entities), 'Expected entity not found.'

test_extract_entities_from_email()