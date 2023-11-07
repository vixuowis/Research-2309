from f00803_get_metadata import *
def test_get_metadata():
    index = {'metadata': {'total_size': 433245184}}
    assert get_metadata(index) == {'total_size': 433245184}
    
    index = {}
    assert get_metadata(index) == {}
    
    index = {'metadata': {'total_size': 0}}
    assert get_metadata(index) == {'total_size': 0}
    
    index = {'metadata': {'total_size': -1}}
    assert get_metadata(index) == {'total_size': -1}
    
    index = {'metadata': {'total_size': 1}}
    assert get_metadata(index) == {'total_size': 1}
