from typing import *
import evaluate

def evaluate(model, eval_dataloader, device):
    """
    Evaluate the model on the evaluation dataset.

    Args:
    - model: The trained model.
    - eval_dataloader: The data loader for the evaluation dataset.
    - device: The device to run the evaluation on.

    Returns:
    The computed metric value.
    """
    metric = evaluate.load("accuracy")
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    return metric.compute()
