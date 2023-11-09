def test_get_legal_document_answer():
    question = 'Who is the licensee in the contract?'
    context = 'We hereby grant the Licensee the exclusive right to develop, construct, operate and promote the Project, as well as to manage the daily operations of the Licensed Facilities during the Term. In consideration for the grant of the License, the Licensee shall pay to the Licensor the full amount of Ten Million (10,000,000) Dollars within thirty (30) days after the execution hereof.'
    answer = get_legal_document_answer(question, context)
    assert 'Licensee' in answer, f'Expected Licensee, but got {answer}'

test_get_legal_document_answer()