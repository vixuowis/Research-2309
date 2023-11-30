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
    
    # Load the tokenizer and model that we will use to generate the sentence embeddings.
    device = torch.device("cuda") if torch.cuda.is_available() else None
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/distilbert-base-nli-stsb-mean-tokens', do_lower_case=True, output_hidden_states=False)
    model = AutoModel.from_pretrained('sentence-transformers/distilbert-base-nli-stsb-mean-tokens')
    
    if device is not None:
        # We can put the model on the GPU to make things faster.
        model = model.cuda()

    sentence_embeddings = []

    for sent in sentences:
        tokenized_sentence = torch.tensor(tokenizer.encode(str(sent)), device=device)
        output = model(tokenized_sentence)
        
        # take the first output embedding and add to sentence embeddings
        sentence_embeddings.append(output[0][0].detach().cpu().numpy())
    
    return torch.Tensor(sentence_embeddings).cpu()

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