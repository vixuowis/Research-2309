from f00100_get_keys import *
def test_get_keys():
	dataset = [{'key1': 'value1', 'key2': 'value2'}, {'key3': 'value3', 'key4': 'value4'}]
	expected_result = ['key1', 'key2']
	assert get_keys(dataset) == expected_result

	dataset = []
	expected_result = []
	assert get_keys(dataset) == expected_result

	dataset = [{'key5': 'value5', 'key6': 'value6'}, {'key7': 'value7', 'key8': 'value8'}, {'key9': 'value9', 'key10': 'value10'}]
	expected_result = ['key5', 'key6']
	assert get_keys(dataset) == expected_result

