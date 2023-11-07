from f00136_training_loop import *
def test_training_loop():
    model = nn.Module()
    train_dataloader = DataLoader()
    num_training_steps = 1000
    num_epochs = 10
    device = 'cuda'
    optimizer = torch.optim.Adam(model.parameters())
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    training_loop(model, train_dataloader, num_training_steps, num_epochs, device, optimizer, lr_scheduler)

test_training_loop()
