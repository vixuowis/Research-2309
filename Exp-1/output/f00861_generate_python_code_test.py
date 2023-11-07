from f00861_generate_python_code import *
def test_generate_python_code():
    src_lang = 'ron_Latn'
    tgt_lang = 'deu_Latn'
    input_code = "Şeful ONU spune că nu există o soluţie militară în Siria"

    translated_code = generate_python_code(src_lang, tgt_lang, input_code)

    assert translated_code == 'UN-Chef sagt, es gibt keine militärische Lösung in Syrien'

    print('Test passed!')

test_generate_python_code()
