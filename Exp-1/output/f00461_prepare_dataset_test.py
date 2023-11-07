from f00461_prepare_dataset import *
def test_prepare_dataset():
    # Test case 1
    example1 = {'input': ' Hello World ', 'output': ' Hello, World! '}
    expected1 = {'input': 'hello world', 'output': 'hello, world!'}
    assert prepare_dataset(example1) == expected1

    # Test case 2
    example2 = {'input': '  foo bar  ', 'output': '  Foo Bar  '}
    expected2 = {'input': 'foo bar', 'output': 'foo bar'}
    assert prepare_dataset(example2) == expected2

    # Test case 3
    example3 = {'input': '  lorem ipsum  ', 'output': '  Lorem Ipsum  '}
    expected3 = {'input': 'lorem ipsum', 'output': 'lorem ipsum'}
    assert prepare_dataset(example3) == expected3

    # Test case 4
    example4 = {'input': '  foo bar  ', 'output': '  Foo Bar  '}
    expected4 = {'input': 'foo bar', 'output': 'foo bar'}
    assert prepare_dataset(example4) == expected4

    # Test case 5
    example5 = {'input': '  lorem ipsum  ', 'output': '  Lorem Ipsum  '}
    expected5 = {'input': 'lorem ipsum', 'output': 'lorem ipsum'}
    assert prepare_dataset(example5) == expected5

