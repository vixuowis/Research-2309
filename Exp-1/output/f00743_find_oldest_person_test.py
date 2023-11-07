from f00743_find_oldest_person import *
def test_find_oldest_person():
	# Test case 1
	document = 'John is 40 years old. Mary is 35 years old. Tom is 50 years old.'
	find_oldest_person(document)

	# Test case 2
	document = 'Alice is 60 years old. Bob is 55 years old. Emma is 70 years old.'
	find_oldest_person(document)

	# Test case 3
	document = 'Sam is 45 years old. Sarah is 42 years old. Michael is 38 years old.'
	find_oldest_person(document)

	# Test case 4
	document = 'Oliver is 25 years old. Olivia is 30 years old. Oscar is 28 years old.'
	find_oldest_person(document)

	# Test case 5
	document = 'Sophia is 20 years old. Ethan is 18 years old. Ava is 22 years old.'
	find_oldest_person(document)


test_find_oldest_person()
