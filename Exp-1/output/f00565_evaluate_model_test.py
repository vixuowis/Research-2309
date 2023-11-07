from f00565_evaluate_model import *
import evaluate
from tqdm import tqdm


def evaluate_model(model, module, val_dataloader):
    """
    Evaluate the model on the validation dataset.

    Args:
        model (AutoModelForObjectDetection): The trained model.
        module: The evaluation module.
        val_dataloader (torch.utils.data.DataLoader): The validation dataloader.
    """
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_dataloader)):
            pixel_values = batch["pixel_values"]
            pixel_mask = batch["pixel_mask"]

            labels = [
                {k: v for k, v in t.items()} for t in batch["labels"]
            ]  # these are in DETR format, resized + normalized

            # forward pass
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

            orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
            results = im_processor.post_process(outputs, orig_target_sizes)  # convert outputs of model to COCO api

            module.add(prediction=results, reference=labels)
            del batch


# Test the evaluate_model function
model = AutoModelForObjectDetection.from_pretrained("devonho/detr-resnet-50_finetuned_cppe5")
module = evaluate.load("ybelkada/cocoevaluate", coco=test_ds_coco_format.coco)
val_dataloader = torch.utils.data.DataLoader(
    test_ds_coco_format, batch_size=8, shuffle=False, num_workers=4, collate_fn=collate_fn
)
evaluate_model(model, module, val_dataloader)
