from transformers import TapexTokenizer, BartForConditionalGeneration
import pandas as pd

def get_olympic_year(city_name):
    '''
    This function takes in a city name and returns the year when the Olympic Games were held in that city.
    It uses the TAPEX (Table Pre-training via Execution) model from Hugging Face Transformers to perform table reasoning.
    '''
    tokenizer = TapexTokenizer.from_pretrained('microsoft/tapex-base')
    model = BartForConditionalGeneration.from_pretrained('microsoft/tapex-base')
    data = {
        'year': [1896, 1900, 1904, 2004, 2008, 2012],
        'city': ['athens', 'paris', 'st. louis', 'athens', 'beijing', 'london']
    }
    table = pd.DataFrame.from_dict(data)
    query = f"select year where city = {city_name}"
    encoding = tokenizer(table=table, query=query, return_tensors='pt')
    outputs = model.generate(**encoding)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)