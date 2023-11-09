from ultralyticsplus import YOLO

def detect_players(image_path):
    """
    Detect the location of players in an image from a Counter-Strike: Global Offensive (CS:GO) game.

    Args:
        image_path (str): The path to the image file.

    Returns:
        detected_players (list): A list of bounding boxes for detected players.
    """
    model = YOLO('keremberke/yolov8n-csgo-player-detection')
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000
    results = model.predict(image_path)
    detected_players = results[0].boxes
    return detected_players