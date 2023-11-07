from typing import *
from typing import List, Dict

def formatted_anns(image_id: int, category: List[int], area: List[float], bbox: List[List[float]]) -> List[Dict]:
    """Reformats annotations for a single example.

    Args:
        image_id (int): The image ID.
        category (List[int]): The category IDs.
        area (List[float]): The areas of the objects.
        bbox (List[List[float]]): The bounding boxes of the objects.

    Returns:
        List[Dict]: The reformatted annotations.
    """
    annotations = []
    for i in range(0, len(category)):
        new_ann = {
            "image_id": image_id,
            "category_id": category[i],
            "isCrowd": 0,
            "area": area[i],
            "bbox": list(bbox[i]),
        }
        annotations.append(new_ann)

    return annotations
