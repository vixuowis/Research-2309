from f00479_convert_id_to_label_name import *
def test_convert_id_to_label_name():
    id2label = {
        '79': 'prime_rib',
        '80': 'chicken_curry',
        '81': 'lobster_bisque'
    }
    assert convert_id_to_label_name(id2label, '79') == 'prime_rib'
    assert convert_id_to_label_name(id2label, '80') == 'chicken_curry'
    assert convert_id_to_label_name(id2label, '81') == 'lobster_bisque'


test_convert_id_to_label_name()
