# function_import --------------------

from transformers import AutoTokenizer, AutoModel
import torch

# function_code --------------------

def generate_sentence_embeddings(sentences):
    """
    Generate sentence embeddings for the given sentences using a pre-trained model.

    Args:
        sentences (list): A list of sentences for which to generate embeddings.

    Returns:
        torch.Tensor: A tensor containing the sentence embeddings.
    """
    
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    with torch.no_grad():
        sentence_embeddings = []
        for sentence in sentences:
            tokenized_sentence = tokenizer(sentence, return_tensors='pt')
            output = model(**tokenized_sentence)  # This line will fail if there are errors.
            
            embeddings = output['last_hidden_state']
            mean_embedding = torch.mean(embeddings[0], dim=0)  # Mean of sentence embedding.
            sentence_embeddings.append(mean_embedding)
        
        sentence_embeddings = torch.stack(sentence_embeddings)
    return sentence_embeddings

# test_function_code --------------------

def test_generate_sentence_embeddings():
    """
    Test the generate_sentence_embeddings function.
    """
    sentences = ['Анализировать текст российской газеты', 'Это просто пример предложения', 'Мы тестируем функцию генерации вложений предложений']
    embeddings = generate_sentence_embeddings(sentences)
    assert embeddings.shape[0] == len(sentences), 'Number of embeddings does not match number of sentences'
    assert embeddings.shape[1] == 1024, 'Embedding size does not match expected size'
    return 'All Tests Passed'


# call_test_function_code --------------------

test_generate_sentence_embeddings()