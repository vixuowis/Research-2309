from f00600_filter_long_documents import *
def test_filter_long_documents():
    dataset = [
        {'words': 'This is a short document.', 'question': 'What is this?'},
        {'words': 'This is a long document that exceeds the maximum length.', 'question': 'What is this?'},
        {'words': 'This is another short document.', 'question': 'What is this?'},
    ]
    filtered_dataset = filter_long_documents(dataset)
    assert len(filtered_dataset) == 2
    assert filtered_dataset[0]['words'] == 'This is a short document.'
    assert filtered_dataset[1]['words'] == 'This is another short document.'
    print('All test cases pass.')

test_filter_long_documents()
