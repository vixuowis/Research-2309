def test_load_and_play_soccer():
    """
    This function tests the load_and_play_soccer function.
    """
    repo_id = 'Raiden-1001/poca-Soccerv7'
    local_dir = './downloads'
    load_and_play_soccer(repo_id, local_dir)
    assert os.path.exists(local_dir+'/SoccerTwos'), 'Model not found in the specified directory'

test_load_and_play_soccer()