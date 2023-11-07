from f00678_generate_caption import *
def test_generate_caption():
    image_url = 'https://images.unsplash.com/photo-1583160247711-2191776b4b91?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=3542&q=80'
    expected_caption = 'A puppy in a flower bed'
    assert generate_caption(image_url) == expected_caption

test_generate_caption()
