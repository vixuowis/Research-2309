from f00118_create_trainer import *
def test_create_trainer():
    model = nn.Module()
    args = TrainingArguments()
    train_dataset = Dataset()
    eval_dataset = Dataset()
    compute_metrics = Callable()
    trainer = create_trainer(model, args, train_dataset, eval_dataset, compute_metrics)
    assert isinstance(trainer, Trainer)


test_create_trainer()

