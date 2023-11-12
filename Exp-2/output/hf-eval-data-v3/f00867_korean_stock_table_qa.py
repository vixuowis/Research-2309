# function_import --------------------

from transformers import pipeline

# function_code --------------------

def korean_stock_table_qa(table: dict, query: str) -> str:
    '''
    This function uses the Hugging Face Transformers pipeline for table-question-answering to find the accurate information in a table related to the Korean stock market.

    Args:
        table (dict): The table containing the Korean stock market data. It should be a dictionary with 'header' and 'rows' as keys.
        query (str): The question to be answered based on the table.

    Returns:
        str: The answer to the query based on the table provided.
    '''
    table_qa = pipeline('table-question-answering', model='dsba-lab/koreapas-finetuned-korwikitq')
    answer = table_qa(table=table, query=query)
    return answer

# test_function_code --------------------

def test_korean_stock_table_qa():
    '''
    This function tests the korean_stock_table_qa function with some test cases.
    '''
    table1 = {'header': ['company', 'stock price', 'market cap'], 'rows': [['samsung', 50000, 100000], ['lg', 30000, 45000]]}
    query1 = 'Which company has a higher market cap?'
    assert korean_stock_table_qa(table1, query1) == 'samsung'

    table2 = {'header': ['company', 'stock price', 'market cap'], 'rows': [['samsung', 30000, 45000], ['lg', 50000, 100000]]}
    query2 = 'Which company has a higher stock price?'
    assert korean_stock_table_qa(table2, query2) == 'lg'

    table3 = {'header': ['company', 'stock price', 'market cap'], 'rows': [['samsung', 50000, 100000], ['lg', 30000, 45000], ['hyundai', 60000, 120000]]}
    query3 = 'Which company has the highest stock price?'
    assert korean_stock_table_qa(table3, query3) == 'hyundai'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_korean_stock_table_qa()