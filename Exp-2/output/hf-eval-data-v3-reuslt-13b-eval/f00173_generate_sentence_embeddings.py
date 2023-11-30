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
    
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2")    

    tokens = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt", max_length=100)
    
    with torch.no_grad():
        model_outputs = model(**tokens)        
        
    embeddings = torch.mean(model_outputs[0], dim=-2)
    return torch.squeeze(embeddings, 1)

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