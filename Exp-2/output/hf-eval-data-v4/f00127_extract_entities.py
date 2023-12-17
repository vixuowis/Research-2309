# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# function_code --------------------

def extract_entities(news_article):
    """
    Extracts named entities such as people, organizations, and locations from a given news article using a pre-trained NER model.

    Parameters:
    news_article (str): A string containing the text of the news article.

    Returns:
    list: A list of dictionaries with entity name and entity type.
    """
    tokenizer = AutoTokenizer.from_pretrained('Davlan/distilbert-base-multilingual-cased-ner-hrl')
    model = AutoModelForTokenClassification.from_pretrained('Davlan/distilbert-base-multilingual-cased-ner-hrl')
    nlp = pipeline('ner', model=model, tokenizer=tokenizer)

    ner_results = nlp(news_article)
    return ner_results

# test_function_code --------------------

def test_extract_entities():
    print("Testing started.")

    # 测试用例 1：测试新闻文章包含人名、组织和地点
    print("Testing case [1/1] started.")
    news_article = "Nader Jokhadar had given Syria the lead with a well-struck header in the seventh minute."
    expected_entities = [{'entity_group': 'PER', 'score': ..., 'word': 'Nader Jokhadar', 'start': ..., 'end': ...},
                         {'entity_group': 'LOC', 'score': ..., 'word': 'Syria', 'start': ..., 'end': ...}]
    actual_entities = extract_entities(news_article)

    # 检查是否识别出了人名和地点
    assert any(ent['entity_group'] == 'PER' for ent in actual_entities), "Test case [1/1] failed: Person entity not found."
    assert any(ent['entity_group'] == 'LOC' for ent in actual_entities), "Test case [1/1] failed: Location entity not found."
    
    print("Testing finished.")

# 运行测试函数
test_extract_entities()