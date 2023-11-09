def test_predict_pokemon_hp():
    """
    This function tests the predict_pokemon_hp function.
    It uses a sample Pokemon's attributes and checks if the predicted HP is within a reasonable range.
    """
    # Define a sample Pokemon's attributes
    sample_pokemon = {'name': 'Pikachu', 'type1': 'Electric', 'type2': '', 'total': 320, 'hp': 35, 'attack': 55, 'defense': 40, 'sp_atk': 50, 'sp_def': 50, 'speed': 90, 'generation': 1, 'legendary': False}
    
    # Predict the HP of the sample Pokemon
    predicted_hp = predict_pokemon_hp(sample_pokemon)
    
    # Check if the predicted HP is within a reasonable range
    assert 0 <= predicted_hp <= 250, 'The predicted HP is out of range.'

test_predict_pokemon_hp()