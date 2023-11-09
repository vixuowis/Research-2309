def test_get_software_engineers_hired_in_2020():
    """
    This function tests the get_software_engineers_hired_in_2020 function.
    """
    employee_data = {
        'name': ['John Doe', 'Jane Doe', 'Alice', 'Bob'],
        'title': ['Software Engineer', 'Software Engineer', 'Data Scientist', 'Product Manager'],
        'department': ['Engineering', 'Engineering', 'Data', 'Product'],
        'hire_date': ['2020-01-01', '2019-01-01', '2020-01-01', '2020-01-01']
    }
    result = get_software_engineers_hired_in_2020(employee_data)
    assert len(result) == 1 and 'John Doe' in result, 'test_get_software_engineers_hired_in_2020 failed.'

test_get_software_engineers_hired_in_2020()