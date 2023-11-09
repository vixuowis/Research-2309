from ultralyticsplus import YOLO, render_result


def detect_csgo_players(game_image):
    """
    Detects players in a live game of Counter-Strike: Global Offensive (CS:GO) using a pre-trained YOLO model.

    Args:
        game_image (str): Path to the game screen image.

    Returns:
        A rendered image with detected players' bounding boxes.
    """
    model = YOLO('keremberke/yolov8m-csgo-player-detection')
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000
    results = model.predict(game_image)
    print(results[0].boxes)
    render = render_result(model=model, image=game_image, result=results[0])
    return render