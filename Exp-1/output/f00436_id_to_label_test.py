from f00436_id_to_label import *
def test_id_to_label():
    assert id_to_label('2') == 'app_error'
    assert id_to_label('5') == 'server_error'
    assert id_to_label('8') == 'unknown_error'
    assert id_to_label('10') == 'invalid_input'
    assert id_to_label('12') == 'out_of_memory'
