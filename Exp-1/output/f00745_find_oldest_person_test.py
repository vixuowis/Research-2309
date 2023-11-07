from f00745_find_oldest_person import *
def test_find_oldest_person():
	# Test case 1
	document = "This is a document about people. John is 30 years old and Mary is 40 years old."
	expected = "Mary"
	assert find_oldest_person(document) == expected

	# Test case 2
	document = "The oldest person in this document is Alice."
	expected = "Alice"
	assert find_oldest_person(document) == expected

	# Test case 3
	document = "There are no people in this document."
	expected = None
	assert find_oldest_person(document) == expected

	# Test case 4
	document = "John and Mary are both 50 years old."
	expected = "John and Mary"
	assert find_oldest_person(document) == expected

	# Test case 5
	document = "The oldest person in this document is Bob."
	expected = "Bob"
	assert find_oldest_person(document) == expected


test_find_oldest_person()
