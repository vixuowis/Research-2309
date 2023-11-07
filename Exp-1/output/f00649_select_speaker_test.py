from f00649_select_speaker import *
def test_select_speaker():
    assert select_speaker(1) == True
    assert select_speaker(2) == False
    assert select_speaker(3) == True
    assert select_speaker(4) == False
    assert select_speaker(5) == True

test_select_speaker()
