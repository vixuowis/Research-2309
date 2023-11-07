from f00651_get_remaining_examples import *
def test_get_remaining_examples():
	assert get_remaining_examples([]) == 0
	assert get_remaining_examples([1, 2, 3]) == 3
	assert get_remaining_examples(["a", "b", "c", "d", "e"]) == 5


if __name__ == "__main__":
	test_get_remaining_examples()
