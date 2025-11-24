import json
from typing import List, Set, Dict
from PIL import Image, ImageDraw

class Coordinate():
    def __init__(self, x, y):
        self.x = x
        self.y = y

class BoundingBox:
    def __init__(self, y_min: float, x_min: float, y_max: float, x_max: float):
        self.y_min = y_min
        self.x_min = x_min
        self.y_max = y_max
        self.x_max = x_max
        self.top_left = Coordinate(x_min, y_min)
        self.top_right = Coordinate(x_max, y_min)
        self.bottom_left = Coordinate(x_min, y_max)
        self.bottom_right = Coordinate(x_max, y_max)

    def __str__(self):
        return f"BoundingBox(y_min={self.y_min}, x_min={self.x_min}, y_max={self.y_max}, x_max={self.x_max})"

    def __repr__(self):
        return self.__str__()

    def get_area(self) -> float:
        width = self.x_max - self.x_min
        height = self.y_max - self.y_min
        return abs(width) * abs(height)

    def get_intersection_area(self, other: 'BoundingBox') -> float:
        """Calculates the area of overlap between two bounding boxes."""

        # Determine the coordinates of the intersection box
        # For x-coordinates: the intersection's left edge is the MAX of the two left edges
        inter_x_min = max(self.x_min, other.x_min)
        # The intersection's right edge is the MIN of the two right edges
        inter_x_max = min(self.x_max, other.x_max)

        # For y-coordinates (remember y_max is usually the bottom):
        inter_y_min = max(self.y_min, other.y_min)
        inter_y_max = min(self.y_max, other.y_max)

        # Calculate width and height of the intersection
        inter_width = max(0.0, inter_x_max - inter_x_min)
        inter_height = max(0.0, inter_y_max - inter_y_min)

        return inter_width * inter_height

    def intersection_over_self_area(self, other: 'BoundingBox') -> float:
        """
        Calculates the percentage of this bounding box's area that is covered
        by the 'other' bounding box (IoA).
        Returns a float between 0.0 and 1.0.
        """
        self_area = self.get_area()

        # If the box has zero area (e.g., it's a line or a point), avoid division by zero.
        if self_area == 0.0:
            return 0.0

        intersection_area = self.get_intersection_area(other)

        # Calculate the ratio (IoA)
        return intersection_area / self_area

class Detection:
    def __init__(self, symbol: str, detection_score: float, bounding_box: BoundingBox):
        self.symbol = symbol
        self.detection_score = detection_score
        self.bounding_box = bounding_box

    def __str__(self):
        return f"Detection({self.symbol}, {self.detection_score}, {self.bounding_box})"

    def __repr__(self):
        return self.__str__()

    def find_detection_with_biggest_self_intersection(self, detections: List["Detection"]) -> "Detection":
        biggest_intersection = 0
        best_detection = None
        for i in range(len(detections)):
            detection = detections[i]
            intersection = self.bounding_box.intersection_over_self_area(detection.bounding_box)
            if intersection > biggest_intersection:
                biggest_intersection = biggest_intersection
                best_detection = detection
        return best_detection

    def total_overlap_with_others(self, detections: List["Detection"]) -> float:


class Quarter(Detection):
    def __init__(self, detection_score: float, bounding_box: BoundingBox, children: Set["Symbol"] = None):
        super().__init__("quarter", detection_score, bounding_box)
        self.children = children if children else {}
        self._up, self._down, self._right, self._left = [None] * 4

    @property
    def up(self):
        return self._up

    @up.setter
    def up(self, other: "Quarter"):
        self._up = other
        other._down = self

    @property
    def down(self):
        return self._down

    @down.setter
    def down(self, other: "Quarter"):
        self._down = other
        other._up = self

    @property
    def right(self):
        return self._right

    @right.setter
    def right(self, other: "Quarter"):
        self._right = other
        other._left = self

    @property
    def left(self):
        return self._left

    @left.setter
    def left(self, other: "Quarter"):
        self._left = other
        other._right = self

    def __eq__(self, other):
        if not isinstance(other, Quarter):
            return False

        return (self.children == other.children and
                self.up is other.up and
                self.down is other.down and
                self.right is other.right and
                self.left is other.left)

    def __hash__(self):
        return hash((
            tuple(self.children),
            id(self.up),
            id(self.down),
            id(self.right),
            id(self.left)
        ))

class Symbol(Detection):
    def __init__(self, symbol: str, detection_score: float, bounding_box: BoundingBox):
        super().__init__(symbol, detection_score, bounding_box)

    def add_parent_children_relationship_to_best_quarter(self, quarters: List["Quarter"]):
        best_quarter = self.find_detection_with_biggest_self_intersection(quarters)
        best_quarter.children.add(self)

    def __eq__(self, other):
        if not isinstance(other, Symbol):
            return False

        return self.symbol == other.symbol

    def __hash__(self):
        return hash(self.symbol)

    @staticmethod
    def of(symbol: str) -> "Symbol":
        return Symbol(symbol, None, None)

class Board:

    def __init__(self, detections: List[Detection]):
        self.original_detections = detections.copy()
        self.detections = detections.copy()

    @staticmethod
    def from_predictions(response: dict) -> "Board":
        predictions = response["predictions"][0]
        num_detections = int(predictions["num_detections"])
        symbols = predictions["detection_classes_as_text"]
        scores = predictions["detection_scores"]
        boxes = predictions["detection_boxes"]
        bounding_boxes = [BoundingBox(*box) for box in boxes]
        return Board([Detection(symbols[i], scores[i], bounding_boxes[i]) for i in range(num_detections)])

    def filter_by_confidence_score(self, absolut_score: float = None, symbol_scores: Dict[str, float] = None):
        filtered_detections = []
        for detection in self.detections:
            limit = absolut_score
            symbol = detection.symbol
            if symbol_scores and symbol in symbol_scores:
                limit = symbol_scores[symbol]
            if not limit:
                raise ValueError(f"{symbol} has no minimum confidence score.")
            if detection.detection_score >= limit:
                filtered_detections.append(detection)
        self.detections = filtered_detections





def parse_predictions(response: dict) -> List[Detection]:
    predictions = response["predictions"][0]
    num_detections = int(predictions["num_detections"])
    symbols = predictions["detection_classes_as_text"]
    scores = predictions["detection_scores"]
    boxes = predictions["detection_boxes"]
    bounding_boxes = [BoundingBox(*box) for box in boxes]
    return [Detection(symbols[i], scores[i], bounding_boxes[i]) for i in range(num_detections)]

def visualize_detections(detections: List[Detection], img_width: int = 512, img_height: int = 512):
    """
    Visualizes a list of Detection objects on a blank canvas and shows the result.
    """
    # Create a white canvas
    img = Image.new('RGB', (img_width, img_height), color='white')
    draw = ImageDraw.Draw(img)

    for detection in detections:
        # 1. De-normalize coordinates (convert 0-1 range to pixels)
        box = detection.bounding_box

        # PIL expects [left, top, right, bottom] (x_min, y_min, x_max, y_max)
        left = box.x_min * img_width
        top = box.y_min * img_height
        right = box.x_max * img_width
        bottom = box.y_max * img_height

        # 2. Determine Color (Black for 'quarter', Green for others)
        color = "black" if detection.symbol == 'quarter' else "green"

        # 3. Draw the Bounding Box
        draw.rectangle([left, top, right, bottom], outline=color, width=3)

        # 4. Draw Label
        label_text = f"{detection.symbol}: {detection.detection_score:.2f}"

        # Get text size for the background box
        text_bbox = draw.textbbox((left, top), label_text)
        # Draw filled box behind text for readability
        draw.rectangle(text_bbox, fill=color)
        # Draw text
        draw.text((left, top), label_text, fill="white")

    # Show the image using system viewer
    img.show()

if __name__ == '__main__':

    raw_response = {
      "predictions": [
        {
          "key": "some_unique_id_or_filename",
          "detection_classes": [13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 9.0, 4.0, 13.0, 13.0, 2.0, 8.0, 13.0, 4.0, 2.0, 7.0, 11.0, 8.0, 12.0, 13.0, 7.0, 4.0, 12.0, 4.0, 11.0, 9.0, 11.0, 6.0, 12.0, 8.0, 9.0, 9.0, 13.0, 2.0, 12.0, 2.0, 4.0, 9.0, 13.0, 13.0, 13.0, 13.0, 13.0, 4.0, 4.0, 13.0, 2.0, 13.0, 2.0, 13.0, 8.0, 12.0, 4.0, 9.0, 9.0, 2.0, 5.0, 1.0, 13.0, 5.0, 13.0],
          "num_detections": 61.0,
          "detection_boxes": [[0.687571287, 0.356351644, 0.869298, 0.578866065], [0.142200246, 0.480215549, 0.320300102, 0.703964949], [0.629792929, 0.147475302, 0.817703, 0.366261125], [0.318933487, 0.451581061, 0.499012649, 0.668484747], [0.455426931, 0.184659898, 0.6385988, 0.407022178], [0.365660965, 0.660142422, 0.547049463, 0.877313256], [0.554288089, 0.28592, 0.647891819, 0.387712747], [0.187457725, 0.540149868, 0.280982196, 0.643518388], [0.284427971, 0.237666592, 0.472416192, 0.46134609], [0.538045287, 0.613420308, 0.715733171, 0.83563], [0.376199067, 0.774440289, 0.469352543, 0.877013683], [0.29016602, 0.304236561, 0.384220719, 0.407631189], [0.504032671, 0.402367473, 0.684030592, 0.615810752], [0.547924221, 0.453511924, 0.642777503, 0.559712291], [0.386566401, 0.345520288, 0.480104685, 0.44859305], [0.369489312, 0.242243588, 0.463357329, 0.34576261], [0.526126862, 0.184115544, 0.623480201, 0.290217638], [0.731468141, 0.242141277, 0.828969538, 0.348475724], [0.313234895, 0.464894772, 0.407191, 0.568612576], [0.112947598, 0.271577775, 0.295213282, 0.491630912], [0.415576, 0.556017756, 0.507944, 0.660677314], [0.451210082, 0.704885244, 0.544997454, 0.807373166], [0.730015278, 0.414453566, 0.822258, 0.51885587], [0.356134772, 0.672748208, 0.450801075, 0.775638461], [0.706528425, 0.141603202, 0.800549626, 0.24625802], [0.454897791, 0.262046903, 0.546900034, 0.364778608], [0.157549798, 0.324621439, 0.250845462, 0.428755462], [0.633032262, 0.218501821, 0.729953, 0.319634199], [0.583359778, 0.67250216, 0.674671829, 0.77561748], [0.281843126, 0.275688589, 0.417380273, 0.43672955], [0.489511669, 0.277642936, 0.584705174, 0.380222231], [0.373589724, 0.241535738, 0.460708112, 0.344675958], [0.281843126, 0.275688589, 0.417380273, 0.43672955], [0.0316170678, 0.805090904, 0.141997576, 1.0], [0.685199738, 0.359874517, 0.861468792, 0.574326098], [0.834466159, 0.819572, 0.992513239, 1.0], [0.150642037, 0.486594409, 0.321294367, 0.700012147], [0.421788663, 0.555024743, 0.505609691, 0.658911824], [0.376024455, 0.226985827, 0.565938711, 0.448396862], [0.179318324, 0.245900393, 0.376460433, 0.463202238], [0.195679456, 0.472280353, 0.406635791, 0.693548381], [0.554153919, 0.266331315, 0.650433421, 0.370691836], [0.53685683, 0.251951694, 0.634685218, 0.354381621], [0.403009802, 0.685286224, 0.508426249, 0.792055905], [0.427030385, 0.709615111, 0.523829043, 0.832435369], [0.724583626, 0.234725326, 0.826208711, 0.345998079], [0.396423072, 0.324908495, 0.491901189, 0.426998258], [0.503710389, 0.2454395, 0.653733969, 0.418763131], [0.899859369, 0.0, 1.0, 0.227638602], [0.155552953, 0.458509803, 0.304822326, 0.651194811], [0.362503946, 0.673709631, 0.448435068, 0.778879166], [0.538399637, 0.616558611, 0.71427232, 0.834125], [0.504161358, 0.397488773, 0.681790471, 0.617040455], [0.43554008, 0.285422862, 0.521507442, 0.398500443], [0.486469775, 0.23714757, 0.57841444, 0.361984551], [0.0338350683, 0.0182977617, 0.139664844, 0.219830707], [0.189971343, 0.541429, 0.274459481, 0.645003736], [0.421788663, 0.555024743, 0.505609691, 0.658911824], [0.329071075, 0.223478973, 0.4862248, 0.405014873], [0.362503946, 0.673709631, 0.448435068, 0.778879166], [0.287748605, 0.301342785, 0.385171801, 0.408741653]],
          "detection_scores": [0.977873862, 0.975723624, 0.975592256, 0.973305702, 0.971991301, 0.971522212, 0.969656765, 0.968496382, 0.966274619, 0.966195643, 0.962615848, 0.960797191, 0.957286239, 0.955471933, 0.954563498, 0.954544067, 0.953391492, 0.948669076, 0.946552515, 0.946460605, 0.94468677, 0.943299949, 0.935334265, 0.931677878, 0.922740221, 0.920787275, 0.910910189, 0.899996102, 0.887703836, 0.171139255, 0.170247465, 0.142530978, 0.136380777, 0.125907093, 0.122196302, 0.1093603, 0.0982662216, 0.0976125598, 0.0962307602, 0.0851648673, 0.0849650055, 0.084423624, 0.0814789459, 0.0812847316, 0.0794905201, 0.0785956, 0.0781260729, 0.0756918341, 0.0752068684, 0.0746509805, 0.0744592398, 0.0733178407, 0.0714759454, 0.0674561784, 0.0653745532, 0.0643252209, 0.0640641153, 0.0638663322, 0.0634303, 0.0585736074, 0.0568496883],
          "image_info": [512, 512, 1, 0, 512, 512],
          "detection_classes_as_text": ["quarter", "quarter", "quarter", "quarter", "quarter", "quarter", "map", "rat", "quarter", "quarter", "coin", "anchor", "quarter", "rat", "coin", "rum", "spyglass", "anchor", "shark", "quarter", "rum", "rat", "shark", "rat", "spyglass", "map", "spyglass", "parrot", "shark", "anchor", "map", "map", "quarter", "coin", "shark", "coin", "rat", "map", "quarter", "quarter", "quarter", "quarter", "quarter", "rat", "rat", "quarter", "coin", "quarter", "coin", "quarter", "anchor", "shark", "rat", "map", "map", "coin", "arrow_left", "arrow_right", "quarter", "arrow_left", "quarter"]
        }
      ]
    }
    detections = parse_predictions(raw_response)
    filtered_detections = [detection for detection in detections if detection.detection_score > 0.8]
    quarters = [symbol for symbol in filtered_detections if symbol.symbol == "quarter"]
    symbols = [symbol for symbol in filtered_detections if symbol.symbol != "quarter"]
    for symbol in symbols:
        symbol.add_parent_children_relationship_to_best_quarter(quarters)

    # for quarter in quarters:
    #     print(f"{quarter.short_repr()}\n")

    for quarter in quarters:
        print(f"{quarter.bounding_box}")

    visualize_detections(filtered_detections)
