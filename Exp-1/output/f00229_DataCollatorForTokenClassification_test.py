from f00229_DataCollatorForTokenClassification import *
def test_DataCollatorForTokenClassification():
    tokenizer = PreTrainedTokenizer()
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    # Test case 1
    dataset1 = TokenClassificationDataset()
    dataset2 = TokenClassificationDataset()
    datasets = [dataset1, dataset2]
    result = data_collator(datasets)
    assert result == expected_result1, f"Test case 1 failed: {result}"
    
    # Test case 2
    dataset3 = TokenClassificationDataset()
    datasets = [dataset3]
    result = data_collator(datasets)
    assert result == expected_result2, f"Test case 2 failed: {result}"
    
    # Test case 3
    dataset4 = TokenClassificationDataset()
    dataset5 = TokenClassificationDataset()
    dataset6 = TokenClassificationDataset()
    datasets = [dataset4, dataset5, dataset6]
    result = data_collator(datasets)
    assert result == expected_result3, f"Test case 3 failed: {result}"
    
    print("All test cases passed!")


test_DataCollatorForTokenClassification()
