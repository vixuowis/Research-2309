def test_analyze_invoice():
    # Test the analyze_invoice function
    # The function is tested with a sample invoice image and a set of questions
    # The expected answers are known beforehand
    # The function's output is compared with the expected answers using the assert statement
    
    # Test 1
    image_path = 'https://templates.invoicehome.com/invoice-template-us-neat-750px.png'
    question = 'What is the invoice number?'
    expected_answer = '123456'
    assert analyze_invoice(image_path, question) == expected_answer
    
    # Test 2
    image_path = 'https://miro.medium.com/max/787/1*iECQRIiOGTmEFLdWkVIH2g.jpeg'
    question = 'What is the purchase amount?'
    expected_answer = '$100.00'
    assert analyze_invoice(image_path, question) == expected_answer
    
    # Test 3
    image_path = 'https://www.accountingcoach.com/wp-content/uploads/2013/10/income-statement-example@2x.png'
    question = 'What are the 2020 net sales?'
    expected_answer = '$500,000'
    assert analyze_invoice(image_path, question) == expected_answer