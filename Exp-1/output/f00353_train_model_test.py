from f00353_train_model import *
model = PreTrainedModel()
train_dataset = Dataset()
eval_dataset = Dataset()
tokenizer = PreTrainedTokenizer()
data_collator = DataCollator()
compute_metrics = Callable()

train_model(model, train_dataset, eval_dataset, tokenizer, data_collator, compute_metrics)
