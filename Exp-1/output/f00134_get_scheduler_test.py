from f00134_get_scheduler import *
def test_get_scheduler():
    from transformers import AdamW
    import torch

    optimizer = AdamW(torch.nn.Linear(10, 2).parameters())
    num_warmup_steps = 0
    num_training_steps = 1000

    scheduler = get_scheduler('linear', optimizer, num_warmup_steps, num_training_steps)
    assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)

    scheduler = get_scheduler('cosine', optimizer, num_warmup_steps, num_training_steps)
    assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)

    scheduler = get_scheduler('cosine_with_restarts', optimizer, num_warmup_steps, num_training_steps)
    assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)

    scheduler = get_scheduler('polynomial', optimizer, num_warmup_steps, num_training_steps)
    assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)

    scheduler = get_scheduler('constant', optimizer, num_warmup_steps)
    assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)

    print('All tests passed!')

test_get_scheduler()
