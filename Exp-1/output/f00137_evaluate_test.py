from f00137_evaluate import *
import torch

def test_evaluate():
    model = Model()
    eval_dataloader = DataLoader(eval_dataset, batch_size=32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metric_value = evaluate(model, eval_dataloader, device)
    assert isinstance(metric_value, float)
    assert metric_value >= 0.0

    print("Evaluation metric value:", metric_value)


test_evaluate()
