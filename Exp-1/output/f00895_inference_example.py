from typing import *
def inference_example():
    # Example of performing inference using TapasForQuestionAnswering model
    # Returns the predicted answers and aggregations for a given set of queries and table
    # Requires TapasTokenizer and TapasForQuestionAnswering from transformers library
    # Requires pandas library for creating and manipulating tables
    
    model_name = "google/tapas-base-finetuned-wtq"
    model = TapasForQuestionAnswering.from_pretrained(model_name)
    tokenizer = TapasTokenizer.from_pretrained(model_name)
    
    data = {"Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"], "Number of movies": ["87", "53", "69"]}
    queries = [
        "What is the name of the first actor?",
        "How many movies has George Clooney played in?",
        "What is the total number of movies?",
    ]
    table = pd.DataFrame.from_dict(data)
    inputs = tokenizer(table=table, queries=queries, padding="max_length", return_tensors="pt")
    outputs = model(**inputs)
    predicted_answer_coordinates, predicted_aggregation_indices = tokenizer.convert_logits_to_predictions(
        inputs, outputs.logits.detach(), outputs.logits_aggregation.detach()
    )
    
    id2aggregation = {0: "NONE", 1: "SUM", 2: "AVERAGE", 3: "COUNT"}
    aggregation_predictions_string = [id2aggregation[x] for x in predicted_aggregation_indices]
    
    answers = []
    for coordinates in predicted_answer_coordinates:
        if len(coordinates) == 1:
            answers.append(table.iat[coordinates[0]])
        else:
            cell_values = []
            for coordinate in coordinates:
                cell_values.append(table.iat[coordinate])
            answers.append(", ".join(cell_values))
    
    return table, queries, answers, aggregation_predictions_string
