from typing import *
from tqdm.auto import tqdm


def training_loop(model, train_dataloader, num_training_steps, num_epochs, device, optimizer, lr_scheduler):
    """Training loop function

    Args:
        model (nn.Module): The model to be trained
        train_dataloader (DataLoader): The dataloader for training data
        num_training_steps (int): The total number of training steps
        num_epochs (int): The total number of epochs
        device (str): The device to be used for training
        optimizer (torch.optim.Optimizer): The optimizer for updating model parameters
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler

    Returns:
        None
    """
    progress_bar = tqdm(range(num_training_steps))

    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
