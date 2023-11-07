from f00882_step_by_step_document_parsing import *
def test_step_by_step_document_parsing():
    assert step_by_step_document_parsing() == {'menu': {'nm': 'CINNAMON SUGAR', 'unitprice': '17,000', 'cnt': '1 x', 'price': '17,000'}, 'sub_total': {'subtotal_price': '17,000'}, 'total': {'total_price': '17,000', 'cashprice': '20,000', 'changeprice': '3,000'}}
