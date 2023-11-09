from ultralyticsplus import YOLO, render_result


def detect_pcb_defects(image):
    """
    This function detects defects in PCB boards from an image using the YOLO model from Hugging Face Transformers.
    The model is trained on the pcb-defect-segmentation dataset and can detect and segment defects such as Dry_joint, Incorrect_installation, PCB_damage, and Short_circuit.
    
    Parameters:
    image (str): URL or local path to the image
    
    Returns:
    render: Processed image with the detected defects marked
    """
    model = YOLO('keremberke/yolov8m-pcb-defect-segmentation')
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000
    results = model.predict(image)
    render = render_result(model=model, image=image, result=results[0])
    return render