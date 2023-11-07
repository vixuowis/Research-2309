from f00006_preprocess_dataset import *
dataset = load_dataset("common_voice", "fr", split="validation")
preprocessed_dataset = preprocess_dataset(dataset)
print(preprocessed_dataset)
