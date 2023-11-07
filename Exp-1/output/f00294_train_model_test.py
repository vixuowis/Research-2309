from f00294_train_model import *
model = MyModel()
train_dataset = MyDataset(train_data)
eval_dataset = MyDataset(eval_data)

train_model(model, train_dataset, eval_dataset)
