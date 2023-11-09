def test_get_olympic_host_year():
    '''
    This function tests the 'get_olympic_host_year' function with a sample dataset and query.
    '''
    data = {
        'year': [1896, 1900, 1904, 2004, 2008, 2012],
        'city': ['Athens', 'Paris', 'St. Louis', 'Athens', 'Beijing', 'London']
    }
    query = 'Select the year when Beijing hosted the Olympic games'
    assert get_olympic_host_year(data, query) == '2008'