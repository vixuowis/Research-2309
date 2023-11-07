from f00779_create_trainer import *
def test_create_trainer():
    model = MyModel()
    training_args = TrainingArgs()
    train_dataset = MyDataset()
    optimizers = (adam_bnb_optim, None)
    trainer = create_trainer(model, training_args, train_dataset, optimizers)
    assert isinstance(trainer, Trainer)
    assert trainer.model == model
    assert trainer.args == training_args
    assert trainer.train_dataset == train_dataset
    assert trainer.optimizers == optimizers

test_create_trainer()
