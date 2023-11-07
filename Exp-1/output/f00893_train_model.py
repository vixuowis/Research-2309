from typing import *
from transformers import TapasConfig, TapasForQuestionAnswering, AdamW

def train_model(train_dataloader):
    # Train the TapasForQuestionAnswering model
    config = TapasConfig(
        num_aggregation_labels=4,
        use_answer_as_supervision=True,
        answer_loss_cutoff=0.664694,
        cell_selection_preference=0.207951,
        huber_loss_delta=0.121194,
        init_cell_selection_weights_to_zero=True,
        select_one_column=True,
        allow_empty_column_selection=False,
        temperature=0.0352513,
    )
    model = TapasForQuestionAnswering.from_pretrained("google/tapas-base", config=config)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    model.train()
    for epoch in range(2):
        for batch in train_dataloader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            token_type_ids = batch["token_type_ids"]
            labels = batch["labels"]
            numeric_values = batch["numeric_values"]
            numeric_values_scale = batch["numeric_values_scale"]
            float_answer = batch["float_answer"]

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
                numeric_values=numeric_values,
                numeric_values_scale=numeric_values_scale,
                float_answer=float_answer,
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
