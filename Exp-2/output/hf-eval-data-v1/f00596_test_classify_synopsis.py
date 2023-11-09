def test_classify_synopsis():
    '''
    This function tests the classify_synopsis function.
    It uses a sample synopsis and checks if the output is one of the expected categories.
    '''
    # Define a sample synopsis
    synopsis = 'Letzte Woche gab es einen Selbstmord in einer nahe gelegenen kolonie'
    
    # Call the classify_synopsis function
    result = classify_synopsis(synopsis)
    
    # Check if the output is one of the expected categories
    assert result in ['Verbrechen', 'Tragödie', 'Stehlen'], f'Error: {result} not in expected categories'
    
    print('All tests passed.')

test_classify_synopsis()