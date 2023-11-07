from f00379_DataCollatorForSeq2Seq import *
import pytest

@pytest.fixture

def test_data_collator():
	data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint, return_tensors="tf")
	assert data_collator.tokenizer == tokenizer
	assert data_collator.model == checkpoint
	assert data_collator.return_tensors == "tf"


def test_data_collator_with_different_return_tensors():
	data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint, return_tensors="pt")
	assert data_collator.tokenizer == tokenizer
	assert data_collator.model == checkpoint
	assert data_collator.return_tensors == "pt"


def test_data_collator_with_default_return_tensors():
	data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)
	assert data_collator.tokenizer == tokenizer
	assert data_collator.model == checkpoint
	assert data_collator.return_tensors == "tf"


if __name__ == '__main__':
	pytest.main([__file__])
