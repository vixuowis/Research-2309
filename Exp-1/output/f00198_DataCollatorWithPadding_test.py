from f00198_DataCollatorWithPadding import *
def test_data_collator():
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")
    assert isinstance(data_collator, DataCollatorWithPadding)


test_data_collator()
