from transformers import TapexTokenizer, BartForConditionalGeneration
import pandas as pd

def get_olympic_host_year(data, query):
    '''
    This function uses the pre-trained model 'microsoft/tapex-base' to answer the query regarding historical Olympic host cities.
    It takes a dictionary of data and a query as input, and returns the answer as output.
    '''
    tokenizer = TapexTokenizer.from_pretrained('microsoft/tapex-base')
    model = BartForConditionalGeneration.from_pretrained('microsoft/tapex-base')
    table = pd.DataFrame.from_dict(data)
    encoding = tokenizer(table=table, query=query, return_tensors='pt')
    outputs = model.generate(**encoding)
    answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return answer