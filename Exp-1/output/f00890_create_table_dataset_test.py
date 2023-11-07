from f00890_create_table_dataset import *
tsv_path = "your_path_to_the_tsv_file"
table_csv_path = "your_path_to_a_directory_containing_all_csv_files"

train_dataloader = create_table_dataset(tsv_path, table_csv_path, tokenizer)

# Test the train_dataloader
for batch in train_dataloader:
    print(batch)
    break
