from f00662_instantiate_trainer import *
from transformers import TrainingArguments, PreTrainedModel, Dataset, DataCollator, PreTrainedTokenizer

training_args = TrainingArguments()
model = PreTrainedModel()
train_dataset = Dataset()
eval_dataset = Dataset()
data_collator = DataCollator()
tokenizer = PreTrainedTokenizer()

instantiate_trainer(training_args, model, train_dataset, eval_dataset, data_collator, tokenizer)
