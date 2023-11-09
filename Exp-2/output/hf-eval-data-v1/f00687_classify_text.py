from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Function to classify a text message into a category using zero-shot classification
# @param sequence: The text message to classify
# @param candidate_labels: The list of possible categories
# @return: The category that the text message most likely belongs to
def classify_text(sequence: str, candidate_labels: list):
    # Load the pre-trained model and tokenizer
    nli_model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')

    # Initialize a list to store the probabilities of each category
    probs_list = []

    # For each category, construct a hypothesis and tokenize the input sequence and hypothesis
    for label in candidate_labels:
        hypothesis = f'This example is {label}.'
        inputs = tokenizer(sequence, hypothesis, return_tensors='pt', truncation=True)

        # Pass the tokenized input to the model and obtain the logits
        logits = nli_model(**inputs)[0]

        # Convert the logits to label probabilities
        entail_contradiction_logits = logits[:, [0, 2]]
        probs = entail_contradiction_logits.softmax(dim=1)
        prob_label_is_true = probs[:, 1].item()

        # Append the probability of the current category to the list
        probs_list.append(prob_label_is_true)

    # Find the index of the category with the highest probability
    category_index = probs_list.index(max(probs_list))

    # Return the category with the highest probability
    return candidate_labels[category_index]