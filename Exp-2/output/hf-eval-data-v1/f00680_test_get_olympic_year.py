def test_get_olympic_year():
    '''
    This function tests the get_olympic_year function.
    It uses assert to check if the returned year is as expected.
    '''
    assert get_olympic_year('beijing') == '2008'
    assert get_olympic_year('athens') in ['1896', '2004']
    assert get_olympic_year('london') == '2012'
test_get_olympic_year()