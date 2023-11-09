from transformers import TapexTokenizer, BartForConditionalGeneration
import pandas as pd

# Function to answer queries based on the input data using TAPEX
# TAPEX (Table Pre-training via Execution) is a conceptually simple and empirically powerful pre-training approach to empower existing models with table reasoning skills.
def table_based_question_answering(data: dict, query: str) -> str:
    # Load the tokenizer and model
    tokenizer = TapexTokenizer.from_pretrained('microsoft/tapex-base-finetuned-wtq')
    model = BartForConditionalGeneration.from_pretrained('microsoft/tapex-base-finetuned-wtq')

    # Prepare the table using pandas and convert the table to a pandas DataFrame
    table = pd.DataFrame.from_dict(data)

    # Tokenize the input (table and query) and retrieve a tensor
    encoding = tokenizer(table=table, query=query, return_tensors='pt')

    # Generate the output by passing the tensor as input to the model
    outputs = model.generate(**encoding)

    # Decode the output tensor to get the answer in a human-readable format
    answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return answer[0]