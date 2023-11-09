from ultralyticsplus import YOLO, render_result

# Function to detect objects in Valorant game using YOLO model
# The function takes an image of the game frame as input and returns the detected objects
# The function uses the 'keremberke/yolov8m-valorant-detection' model from Hugging Face Transformers
# The model is trained specifically for object detection in Valorant game, including detecting dropped spike, enemy, planted spike, and teammate objects
# The function also visualizes the detected objects in the game frame

def detect_objects_in_valorant(game_frame):
    model = YOLO('keremberke/yolov8m-valorant-detection')
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000
    results = model.predict(game_frame)
    render = render_result(model=model, image=game_frame, result=results[0])
    render.show()
    return results