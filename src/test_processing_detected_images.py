import json
from unittest import TestCase

from src.processing_detected_images import Detection, BoundingBox, visualize_detections, Symbol, Quarter


def parse_generated_annotation(data: dict):
    annotations = data.get("annotations", [])
    processed_annotations = [parse_annotation(annotation) for annotation in annotations]
    return [_ for _ in processed_annotations if _]

def parse_annotation(annotation: dict) -> Detection | None:
    if annotation["visible"]:
        x, y, width, height = annotation["bbox_normalized"]
        bbox = BoundingBox(y, x, y + height, x + width)
        return Detection(annotation["category_name"], 1.0, bbox)
    return None

class TestSymbol(TestCase):

    # Access data just like a standard dictionary
    def test_perfect_scenario(self):
        with open('training_data_generator/other_data/annotations/sample_00000.json', 'r') as file:
            data = json.load(file)
        perfect_data = parse_generated_annotation(data)

        Quarter(None, None, {Symbol.of("spyglass"), Symbol.of("arrow_down")})
        Quarter(None, None, {Symbol.of("kraken")})
        Quarter(None, None, {Symbol.of("spyglass"), Symbol.of("coin"), Symbol.of("map")})
        Quarter(None, None, {Symbol.of("rat"), Symbol.of("rat")})
        Quarter(None, None, {Symbol.of("spyglass"), Symbol.of("anchor"), Symbol.of("rum")})
        Quarter(None, None, {Symbol.of("kraken"), Symbol.of("anchor"), Symbol.of("coin")})
        

        visualize_detections(perfect_data)




