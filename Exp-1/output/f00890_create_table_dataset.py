from typing import *
import torch
import pandas as pd

def create_table_dataset(tsv_path, table_csv_path, tokenizer):
    data = pd.read_csv(tsv_path, sep="\t")

    class TableDataset(torch.utils.data.Dataset):
        def __init__(self, data, tokenizer, table_csv_path):
            self.data = data
            self.tokenizer = tokenizer
            self.table_csv_path = table_csv_path

        def __getitem__(self, idx):
            item = self.data.iloc[idx]
            table = pd.read_csv(self.table_csv_path + item.table_file).astype(str)
            encoding = self.tokenizer(
                table=table,
                queries=item.question,
                answer_coordinates=item.answer_coordinates,
                answer_text=item.answer_text,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            encoding = {key: val.squeeze(0) for key, val in encoding.items()}
            encoding["float_answer"] = torch.tensor(item.float_answer)
            return encoding

        def __len__(self):
            return len(self.data)

    train_dataset = TableDataset(data, tokenizer, table_csv_path)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
    return train_dataloader
