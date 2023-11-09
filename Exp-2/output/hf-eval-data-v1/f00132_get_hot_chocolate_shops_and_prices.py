from transformers import TapasTokenizer, TapasForQuestionAnswering
import pandas as pd

# Function to get the shops that sell hot chocolate and their prices
# Uses the TAPAS model for table question answering
# Input: table - a list of lists representing the table data
#        queries - a list of queries to ask the model
# Output: a dictionary with shop names as keys and their hot chocolate prices as values
def get_hot_chocolate_shops_and_prices(table, queries):
    # Load the TAPAS model
    model = TapasForQuestionAnswering.from_pretrained('google/tapas-mini-finetuned-sqa')
    tokenizer = TapasTokenizer.from_pretrained('google/tapas-mini-finetuned-sqa')

    # Convert the table into a pandas dataframe
    dataframe = pd.DataFrame(table[1:], columns=table[0])

    # Tokenize the table and queries
    inputs = tokenizer(table=dataframe, queries=queries, padding=True, truncation=True, return_tensors="pt")

    # Get the model's output
    outputs = model(**inputs)

    # Parse the output to get the shops and their hot chocolate prices
    answered_shops = [table[row_idx][0] for row_idx in outputs['answer_coordinates'][0][:, 0]]
    hot_chocolate_prices = [table[row_idx][2] for row_idx in outputs['answer_coordinates'][0][:, 0]]

    # Return the result as a dictionary
    return {shop: price for shop, price in zip(answered_shops, hot_chocolate_prices)}