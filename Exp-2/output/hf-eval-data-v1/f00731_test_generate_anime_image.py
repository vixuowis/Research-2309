def test_generate_anime_image():
    '''
    This function tests the generate_anime_image function.
    '''
    prompt = 'anime, masterpiece, high quality, 1girl, solo, long hair, looking at viewer, blush, smile, bangs, blue eyes, skirt, medium breasts, iridescent, gradient, colorful'
    negative_prompt = 'simple background, duplicate, retro style, low quality, lowest quality, 1980s, 1990s, 2000s, 2005 2006 2007 2008 2009 2010 2011 2012 2013, bad anatomy, bad proportions, extra digits, lowres, username, artist name, error, duplicate, watermark, signature, text, extra digit, fewer digits, worst quality, jpeg artifacts, blurry'
    image_path = generate_anime_image(prompt, negative_prompt)
    assert isinstance(image_path, str), 'The function should return a string.'
    assert image_path.endswith('.jpg'), 'The returned string should be a path to a .jpg file.'

test_generate_anime_image()