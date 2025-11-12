import enum
import json
import os
import math
from collections import namedtuple, defaultdict
from typing import List, Dict

import cv2
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt

# Import image effects
from image_effects import apply_camera_effects_to_composite

def debug_show_image(image, title='Debug Image'):
    """Display an image in PyCharm's debug window using matplotlib."""
    # Convert BGR/BGRA to RGB for matplotlib
    if len(image.shape) == 3:
        if image.shape[2] == 4:  # BGRA
            display_img = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:  # BGR
            display_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            display_img = image
    else:
        display_img = image

    plt.figure(figsize=(10, 8))
    plt.imshow(display_img)
    plt.title(title)
    plt.axis('off')
    plt.show()


class Coordinate(namedtuple("Coordinate", ["x", "y"])):
    def __add__(self, other):
        """Add two coordinates element-wise."""
        if isinstance(other, Coordinate):
            return Coordinate(self.x + other.x, self.y + other.y)
        raise TypeError(f"Cannot add Coordinate and {type(other)}")

    def __sub__(self, other):
        """Subtract two coordinates element-wise."""
        if isinstance(other, Coordinate):
            return Coordinate(self.x - other.x, self.y - other.y)
        raise TypeError(f"Cannot subtract {type(other)} from Coordinate")

    def rotate_point(self, center: "Coordinate", rotation_deg: float) -> "Coordinate":
        """Rotate a point around the card's center position."""
        angle_rad = math.radians(-rotation_deg)  # Negative because CV2 rotates clockwise
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        # Translate to origin (relative to card center)
        relative = self - center

        # Rotate
        rotated_x = relative.x * cos_a - relative.y * sin_a
        rotated_y = relative.x * sin_a + relative.y * cos_a

        # Translate back
        return Coordinate(rotated_x, rotated_y) + center

class BoundingBox:
    def __init__(self, symbol_name: str, top_left: Coordinate, bottom_right: Coordinate):
        self.symbol_name = symbol_name
        self.top_left = top_left
        self.bottom_right = bottom_right

    def __add__(self, offset):
        """Add a Coordinate offset to both corners of the bounding box."""
        if isinstance(offset, Coordinate):
            return BoundingBox(
                self.symbol_name,
                self.top_left + offset,
                self.bottom_right + offset
            )
        raise TypeError(f"Cannot add BoundingBox and {type(offset)}")

    def __sub__(self, offset):
        """Subtract a Coordinate offset from both corners of the bounding box."""
        if isinstance(offset, Coordinate):
            return BoundingBox(
                self.symbol_name,
                self.top_left - offset,
                self.bottom_right - offset
            )
        raise TypeError(f"Cannot subtract {type(offset)} from BoundingBox")

    def get_bbox_center(self) -> Coordinate:
        """Calculate the center point of a bounding box."""
        center_x = (self.top_left.x + self.bottom_right.x) / 2
        center_y = (self.top_left.y + self.bottom_right.y) / 2
        return Coordinate(center_x, center_y)

    def get_bbox_size(self) -> Coordinate:
        """Calculate the width and height of a bounding box."""
        width = self.bottom_right.x - self.top_left.x
        height = self.bottom_right.y - self.top_left.y
        return Coordinate(width, height)

    @staticmethod
    def from_center(symbol_name: str, center: Coordinate, size: Coordinate) -> "BoundingBox":
        """Create a bounding box from a center point and size."""
        half_width = size.x / 2
        half_height = size.y / 2

        x1 = int(center.x - half_width)
        y1 = int(center.y - half_height)
        x2 = int(center.x + half_width)
        y2 = int(center.y + half_height)

        return BoundingBox(symbol_name, Coordinate(x1, y1), Coordinate(x2, y2))


class QuarterLocation(enum.Enum):
    TOP_LEFT = 'top_left'
    TOP_RIGHT = 'top_right'
    BOTTOM_LEFT = 'bottom_left'
    BOTTOM_RIGHT = 'bottom_right'

    def get_relative_top_left_position_from_center(self, quarter_size: int) -> Coordinate:
        if self == QuarterLocation.TOP_LEFT:
            return Coordinate(-quarter_size, -quarter_size)
        elif self == QuarterLocation.TOP_RIGHT:
            return Coordinate(0, -quarter_size)
        elif self == QuarterLocation.BOTTOM_LEFT:
            return Coordinate(-quarter_size, 0)
        elif self == QuarterLocation.BOTTOM_RIGHT:
            return Coordinate(0, 0)

    def rotate_90_degrees(self) -> 'QuarterLocation':
        """Rotate the quarter location 90 degrees clockwise."""
        rotation_map = {
            QuarterLocation.TOP_LEFT: QuarterLocation.BOTTOM_LEFT,
            QuarterLocation.TOP_RIGHT: QuarterLocation.TOP_LEFT,
            QuarterLocation.BOTTOM_RIGHT: QuarterLocation.TOP_RIGHT,
            QuarterLocation.BOTTOM_LEFT: QuarterLocation.BOTTOM_RIGHT,
        }
        return rotation_map[self]


class Quarter:
    def __init__(self, location: QuarterLocation, symbols: List[str], visible: bool = True):
        self.location = location
        self.symbols = symbols
        self.visible = visible

    def __str__(self):
        return f"Quarter(location={self.location}, symbols={self.symbols}, visible={self.visible})"

    def __repr__(self):
        return self.__str__()

    def get_relative_bounding_box_from_top_left(self, quarter_size: int) -> List[BoundingBox]:
        padding = 4 # pixels
        padded_size= quarter_size - (2 * padding)

        symbol_size = (quarter_size - 3 * padding) / 2
        num_symbols = len(self.symbols)

        positions = self._get_symbol_positions(num_symbols, padded_size, padding)

        bounding_boxes = []
        for idx, symbol_name in enumerate(self.symbols):
            if idx >= len(positions):
                break

            center_x, center_y = positions[idx]

            x1 = int(center_x - symbol_size / 2)
            y1 = int(center_y - symbol_size / 2)
            x2 = int(center_x + symbol_size / 2)
            y2 = int(center_y + symbol_size / 2)

            bounding_boxes.append(BoundingBox(symbol_name, Coordinate(x1, y1), Coordinate(x2, y2)))

        return bounding_boxes

    def _get_symbol_positions(self, num_symbols, padded_size, padding):
        center_x = padding + padded_size * 0.50
        center_y = padding + padded_size * 0.50
        top_left_x = padding + padded_size * 0.25
        top_left_y = padding + padded_size * 0.25
        top_right_x = padding + padded_size * 0.75
        top_right_y = padding + padded_size * 0.25
        bottom_left_x = padding + padded_size * 0.25
        bottom_left_y = padding + padded_size * 0.75
        bottom_right_x = padding + padded_size * 0.75
        bottom_right_y = padding + padded_size * 0.75
        center_bottom_x = padding + padded_size * 0.50
        center_bottom_y = padding + padded_size * 0.75
        if num_symbols == 1:
            return [Coordinate(center_x, center_y)]
        elif num_symbols == 2:
            return [Coordinate(top_left_x, top_left_y), Coordinate(bottom_right_x, bottom_right_y)]
        elif num_symbols == 3:
            return [Coordinate(top_left_x, top_left_y), Coordinate(top_right_x, top_right_y), Coordinate(center_bottom_x, center_bottom_y)]
        elif num_symbols == 4:
            return [
                Coordinate(top_left_x, top_left_y),
                Coordinate(top_right_x, top_right_y),
                Coordinate(bottom_left_x, bottom_left_y),
                Coordinate(bottom_right_x, bottom_right_y)
            ]
        else:
            raise ValueError(f"Invalid number of symbols: {num_symbols}")

    @staticmethod
    def rotate_arrow_name_90_degrees(arrow_name: str) -> str:
        """
        Rotate an arrow symbol name 90 degrees clockwise.

        Args:
            arrow_name: The arrow symbol name (e.g., "arrow_up", "arrow_down", etc.)

        Returns:
            The rotated arrow name, or the original name if not an arrow
        """
        if not arrow_name.startswith("arrow_"):
            return arrow_name

        arrow_rotation_map = {
            "arrow_up": "arrow_left",
            "arrow_right": "arrow_up",
            "arrow_down": "arrow_right",
            "arrow_left": "arrow_down"
        }

        return arrow_rotation_map.get(arrow_name, arrow_name)

class Card:
    def __init__(self, quarters: List[Quarter], image: ndarray):
        self.original_size = image.shape[0] # Assuming square cards
        self.original_image = image
        self.__padded_image = self.create_padded_image(image)
        self.center_position = Coordinate(self.__padded_image.shape[1] // 2, self.__padded_image.shape[0] // 2)
        self.quarters = quarters
        self.z = 0
        self.rotation = 0  # Track current rotation
        self.bounding_boxes = self._get_bounding_boxes()

    @staticmethod
    def create_padded_image(image: ndarray) -> ndarray:
        original_size = image.shape[0]
        diagonal = int(np.sqrt((original_size ** 2) * 2)) + 10
        expanded = np.zeros((diagonal, diagonal, 4), dtype=np.uint8)
        offset = (diagonal - original_size) // 2
        expanded[offset:offset + original_size, offset:offset + original_size] = image
        return expanded

    @staticmethod
    def from_file(cards_dir: str, card_index: int, json_path: str, card_prefix: str="pirate_card-"):
        image = Card.__load_image(cards_dir, card_index, card_prefix)
        quarters = Card.__load_quarters_from_json(json_path, card_index)
        return Card(quarters, image)

    @staticmethod
    def __load_image(cards_dir: str, card_index: int, card_prefix: str):
        image_path = os.path.join(cards_dir, f"{card_prefix}{str(card_index).zfill(2)}.png")
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Card image not found: {image_path}")
        return image


    @staticmethod
    def __load_quarters_from_json(json_path: str, card_index: int):
        with open(json_path, "r") as f:
            cards_data = json.load(f)
        quarters = cards_data[card_index - 1]['card']['quarters']

        # Create quarters with original locations
        return [Quarter(location=QuarterLocation(item[0]), symbols=item[1]) for item in quarters.items()]

    def _get_bounding_boxes(self) -> Dict[QuarterLocation, List[BoundingBox]]:
        all_boxes = defaultdict(list)
        quarter_size = self.original_size // 2

        for quarter in self.quarters:
            if quarter.visible:
                # Add quarter bounding box
                quarter_top_left_coordinate = quarter.location.get_relative_top_left_position_from_center(quarter_size)
                quarter_bbox = BoundingBox(
                    symbol_name="quarter",
                    top_left=quarter_top_left_coordinate + self.center_position,
                    bottom_right=quarter_top_left_coordinate + self.center_position + Coordinate(quarter_size, quarter_size)
                )
                all_boxes[quarter.location].append(quarter_bbox)

                # Add symbol bounding boxes if quarter has symbols
                if quarter.symbols:
                    bounding_boxes_from_quarter_top_left = quarter.get_relative_bounding_box_from_top_left(quarter_size)
                    for bounding_box in bounding_boxes_from_quarter_top_left:
                        all_boxes[quarter.location].append(bounding_box + quarter_top_left_coordinate + self.center_position)

        return all_boxes

    def offset_relative(self, percent_x: float, percent_y: float):
        dx = int(self.original_size * percent_x)
        dy = int(self.original_size * percent_y)
        self.offset(dx, dy)

    def offset(self, dx: int, dy: int):
        """
        Move the card by the specified offset.
        This translates both the center position and the padded image.

        Args:
            dx: Horizontal offset in pixels
            dy: Vertical offset in pixels
        """
        # Update center position
        self.center_position = Coordinate(self.center_position.x + dx, self.center_position.y + dy)

        # Create translation matrix
        translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])

        # Translate the padded image
        # We need to expand the canvas if the translation moves the image outside current bounds
        height, width = self.__padded_image.shape[:2]
        new_width = width + abs(dx)
        new_height = height + abs(dy)

        # Translate the image
        self.__padded_image = cv2.warpAffine(
            self.__padded_image,
            translation_matrix,
            (new_width, new_height),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )

        # Update bounding boxes to reflect the offset
        self.__offset_bounding_boxes(dx, dy)

    def __offset_bounding_boxes(self, offset_dx: int, offset_dy: int):
        """
        Translate all bounding boxes by the given offset amounts.

        Args:
            offset_dx: Horizontal offset in pixels
            offset_dy: Vertical offset in pixels
        """
        offset = Coordinate(offset_dx, offset_dy)

        offset_boxes = defaultdict(list)

        for quarter_location, boxes in self.bounding_boxes.items():
            for bbox in boxes:
                offset_boxes[quarter_location].append(bbox + offset)

        self.bounding_boxes = dict(offset_boxes)

    def rotate(self, rotation_deg: int, center: Coordinate = None):
        """
        Rotate the card image and update bounding boxes.
        If total rotation exceeds 45 degrees, rotate quarter locations and reset rotation.
        """


        center = center if center is not None else self.center_position

        # Rotate the image
        rot_matrix = cv2.getRotationMatrix2D((center.x, center.y), rotation_deg, 1.0)
        self.__padded_image = cv2.warpAffine(self.__padded_image, rot_matrix,
                                             (self.__padded_image.shape[1], self.__padded_image.shape[0]),
                                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

        # Rotate bounding boxes
        self.__rotate_bounding_boxes(rotation_deg, center)

        # Calculate how many 90-degree rotations to perform
        num_90_rotations = round(rotation_deg / 90)

        # Rotate quarters
        self.__rotate_quarters(num_90_rotations)

    def __rotate_bounding_boxes(self, rotation_deg: float, center: Coordinate = None):
        """
        Rotate all bounding boxes around the card's center point.
        Preserves the original size of the bounding boxes (rotates center point only).
        """
        rotated_boxes = defaultdict(list)

        center = center if center is not None else self.center_position

        for quarter_location, boxes in self.bounding_boxes.items():
            for bbox in boxes:
                bbox_center = bbox.get_bbox_center()
                bbox_size = bbox.get_bbox_size()

                rotated_center = bbox_center.rotate_point(center, rotation_deg)
                rotated_bbox = BoundingBox.from_center(bbox.symbol_name, rotated_center, bbox_size)

                rotated_boxes[quarter_location].append(rotated_bbox)

        self.bounding_boxes = dict(rotated_boxes)

    def __rotate_quarters(self, num_90_rotations: int):
        """
        Rotate quarter locations by the specified number of 90-degree increments.
        Also updates the bounding_boxes dictionary keys to match new quarter locations
        and rotates arrow symbol names.
        """

        new_bounding_boxes = dict()
        for old_location, boxes in self.bounding_boxes.items():
            for box in boxes:
                for _ in range(num_90_rotations):
                    box.symbol_name = Quarter.rotate_arrow_name_90_degrees(box.symbol_name)
            new_location = old_location
            for _ in range(num_90_rotations):
                new_location = new_location.rotate_90_degrees()
            new_bounding_boxes[new_location] = boxes
        self.bounding_boxes = new_bounding_boxes


        for _ in range(num_90_rotations):
            for quarter in self.quarters:
                quarter.location = quarter.location.rotate_90_degrees()
                # Rotate arrow symbols
                quarter.symbols = [Quarter.rotate_arrow_name_90_degrees(symbol) for symbol in quarter.symbols]

    def transform_card_bounding_boxes(self, transformation_matrix: np.ndarray):
        """
        Transform all bounding boxes of a card using the given transformation matrix.

        Args:
            card: Card whose bounding boxes need to be transformed
            transformation_matrix: 3x3 transformation matrix (perspective or affine)
        """
        transformed_boxes = defaultdict(list)

        for quarter_loc, boxes in self.bounding_boxes.items():
            for bbox in boxes:
                # Get the four corners of the bounding box
                corners = np.array([
                    [bbox.top_left.x, bbox.top_left.y, 1],
                    [bbox.bottom_right.x, bbox.top_left.y, 1],
                    [bbox.bottom_right.x, bbox.bottom_right.y, 1],
                    [bbox.top_left.x, bbox.bottom_right.y, 1]
                ], dtype=np.float32).T  # Shape: (3, 4)

                # Perspective transform (3x3 matrix)
                transformed_corners = transformation_matrix @ corners
                # Normalize homogeneous coordinates
                transformed_corners /= transformed_corners[2, :]

                # Extract x and y coordinates
                x_coords = transformed_corners[0, :]
                y_coords = transformed_corners[1, :]

                # Create axis-aligned bounding box from transformed corners
                new_top_left = Coordinate(
                    int(np.min(x_coords)),
                    int(np.min(y_coords))
                )
                new_bottom_right = Coordinate(
                    int(np.max(x_coords)),
                    int(np.max(y_coords))
                )

                # Create new bounding box with same symbol name
                transformed_bbox = BoundingBox(
                    bbox.symbol_name,
                    new_top_left,
                    new_bottom_right
                )
                transformed_boxes[quarter_loc].append(transformed_bbox)

        # Update card's bounding boxes
        self.bounding_boxes = dict(transformed_boxes)

    @property
    def padded_image(self) -> ndarray:
        """Get the padded image (read-only access)."""
        return self.__padded_image

    def __str__(self):
        return f"Card(quarters={self.quarters})"

    def __repr__(self):
        return self.__str__()


class PerspectiveTransform:
    """Represents a perspective transform relative to a unit square."""

    def __init__(self,
                 top_left_offset: Coordinate,
                 top_right_offset: Coordinate,
                 bottom_right_offset: Coordinate,
                 bottom_left_offset: Coordinate):
        """
        Define perspective by how much each corner of a unit square moves.

        Args:
            top_left_offset: Offset from (0, 0)
            top_right_offset: Offset from (1, 0)
            bottom_right_offset: Offset from (1, 1)
            bottom_left_offset: Offset from (0, 1)
        """
        self.top_left_offset = top_left_offset
        self.top_right_offset = top_right_offset
        self.bottom_right_offset = bottom_right_offset
        self.bottom_left_offset = bottom_left_offset

    def to_matrix(self, width: int, height: int) -> np.ndarray:
        """Convert to a 3x3 perspective transformation matrix for given dimensions."""
        # Source: unit square
        src_pts = np.float32([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1]
        ])

        # Destination: unit square + offsets
        dst_pts = np.float32([
            [0 + self.top_left_offset.x, 0 + self.top_left_offset.y],
            [1 + self.top_right_offset.x, 0 + self.top_right_offset.y],
            [1 + self.bottom_right_offset.x, 1 + self.bottom_right_offset.y],
            [0 + self.bottom_left_offset.x, 1 + self.bottom_left_offset.y]
        ])

        # Scale to actual dimensions
        src_pts[:, 0] *= width
        src_pts[:, 1] *= height
        dst_pts[:, 0] *= width
        dst_pts[:, 1] *= height

        return cv2.getPerspectiveTransform(src_pts, dst_pts)

    @staticmethod
    def from_strength(strength: float, width: int, height: int, seed: int = None):
        """
        Create a random perspective transform from a strength parameter.

        Args:
            strength: 0.0 to 1.0, controls how much corners can move
            width: Image width
            height: Image height
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)

        # Maximum offset as fraction of image size
        max_offset = strength * 0.15  # 15% max at strength=1.0

        def random_offset():
            return Coordinate(
                np.random.uniform(-max_offset, max_offset),
                np.random.uniform(-max_offset, max_offset)
            )

        return PerspectiveTransform(
            top_left_offset=random_offset(),
            top_right_offset=random_offset(),
            bottom_right_offset=random_offset(),
            bottom_left_offset=random_offset()
        )

class Board:
    def __init__(self, card_size: int, number_of_cards: int):
        self.cards = []
        self.canvas = self.__create_canvas(card_size, number_of_cards)
        self.center_position = Coordinate(self.canvas.shape[1] // 2, self.canvas.shape[0] // 2)

    @staticmethod
    def __create_canvas(card_size: int, number_of_cards: int):
        diagonal = np.sqrt((card_size ** 2) * 2)
        max_rotated_card_size = int(np.ceil(diagonal)) + 10
        canvas_size = max_rotated_card_size * 2 + (number_of_cards - 1) * max_rotated_card_size
        canvas = np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)
        return canvas

    def add_card(self, card: Card, quarter_location: QuarterLocation=None, to_card_index: int = None, to_quarter:
    QuarterLocation = None, on_top: bool = True):
        if len(self.cards):
            quarter_size = card.original_size // 2
            to_card = self.cards[to_card_index]
            quarter_topleft_coordinate = quarter_location.get_relative_top_left_position_from_center(quarter_size)
            to_quarter_topleft_coordinate = to_quarter.get_relative_top_left_position_from_center(quarter_size)
            offset_coordinate = (to_card.center_position  - card.center_position + to_quarter_topleft_coordinate -
                   quarter_topleft_coordinate)
            card.offset(offset_coordinate.x, offset_coordinate.y)
            card.z = max([c.z for c in self.cards]) + 1 if on_top else min([c.z for c in self.cards]) - 1

        self.cards.append(card)

        # Update visibility for all quarters based on z-coordinate and overlap
        self._update_all_quarter_visibility()

        self._paste_card_onto_canvas(card, on_top=on_top)

    def _update_all_quarter_visibility(self):
        """
        Update visibility for all quarters of all cards based on z-coordinate and spatial overlap.
        A quarter is hidden if it's behind another quarter with higher z-coordinate that overlaps it.
        """
        # Reset all quarters to visible first
        for card in self.cards:
            for quarter in card.quarters:
                quarter.visible = True

        quarter_size = self.cards[0].original_size // 2 if self.cards else 0

        # Check each card's quarters against all other cards
        for card in self.cards:
            for quarter in card.quarters:
                # Get the bounding box of this quarter
                quarter_bbox = self._get_quarter_bounding_box(card, quarter, quarter_size)

                # Check against all other cards
                for other_card in self.cards:
                    if other_card is card:
                        continue

                    # Only check if other card is on top (higher z)
                    if other_card.z <= card.z:
                        continue

                    # Check each quarter of the other card
                    for other_quarter in other_card.quarters:
                        other_quarter_bbox = self._get_quarter_bounding_box(other_card, other_quarter, quarter_size)

                        # Check if quarters overlap
                        if self._bboxes_overlap(quarter_bbox, other_quarter_bbox):
                            # This quarter is hidden by the other quarter
                            quarter.visible = False
                            break

                    if not quarter.visible:
                        break

    def _get_quarter_bounding_box(self, card: Card, quarter: Quarter, quarter_size: int) -> tuple:
        """
        Get the bounding box of a quarter in absolute coordinates.

        Returns:
            Tuple of (x1, y1, x2, y2) representing the quarter's bounding box
        """
        # Get quarter's top-left position relative to card center
        quarter_offset = quarter.location.get_relative_top_left_position_from_center(quarter_size)

        # Convert to absolute coordinates
        x1 = card.center_position.x + quarter_offset.x
        y1 = card.center_position.y + quarter_offset.y
        x2 = x1 + quarter_size
        y2 = y1 + quarter_size

        return (x1, y1, x2, y2)

    def _bboxes_overlap(self, bbox1: tuple, bbox2: tuple) -> bool:
        """
        Check if two bounding boxes overlap.

        Args:
            bbox1: Tuple of (x1, y1, x2, y2)
            bbox2: Tuple of (x1, y1, x2, y2)

        Returns:
            True if the bounding boxes overlap, False otherwise
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Check if one box is to the left of the other
        if x2_1 <= x1_2 or x2_2 <= x1_1:
            return False

        # Check if one box is above the other
        if y2_1 <= y1_2 or y2_2 <= y1_1:
            return False

        # Boxes overlap
        return True

    def _paste_card_onto_canvas(self, card: Card, on_top: bool = True):
        """
        Paste a card onto the canvas by directly overwriting non-transparent pixels.

        Args:
            card: Card to paste onto the canvas
            on_top: If True, paste card on top of existing content.
                   If False, paste card behind existing content.
        """
        card_img = card.padded_image
        card_h, card_w = card_img.shape[:2]

        # Extract the overlapping regions
        canvas_region = self.canvas[0:card_h, 0:card_w]

        # Determine which layer overwrites
        if on_top:
            mask = card_img[:, :, 3] > 0
            canvas_region[mask] = card_img[mask]
        else:
            mask = canvas_region[:, :, 3] == 0
            canvas_region[mask] = card_img[mask]

    def rotate(self, rotation_deg: int, rotate_canvas: bool = True):
        """
        Rotate the entire board by rotating all cards around the board's center.
        Adds padding if rotation would move content outside canvas.

        Args:
            rotation_deg: Rotation angle in degrees (positive = counter-clockwise)
            rotate_canvas: Whether to also rotate the canvas
        """
        if not rotate_canvas:
            # Just rotate cards without expanding canvas
            for card in self.cards:
                card.rotate(rotation_deg, center=self.center_position)
            return

        # Calculate required canvas size after rotation
        h, w = self.canvas.shape[:2]

        # Calculate new dimensions needed to fit rotated content
        angle_rad = abs(math.radians(rotation_deg))
        new_w = int(abs(w * math.cos(angle_rad)) + abs(h * math.sin(angle_rad)))
        new_h = int(abs(w * math.sin(angle_rad)) + abs(h * math.cos(angle_rad)))

        # Calculate padding needed
        pad_w = max(0, (new_w - w) // 2)
        pad_h = max(0, (new_h - h) // 2)

        if pad_w > 0 or pad_h > 0:
            print(f"[Board.rotate] Expanding canvas for {rotation_deg}° rotation:")
            print(f"  Original size: {w}x{h}")
            print(f"  Required size: {new_w}x{new_h}")
            print(f"  Adding padding: horizontal={pad_w}px, vertical={pad_h}px")
            # Expand canvas with padding
            self._add_padding(pad_w, pad_h)
            print(f"  New canvas size: {self.canvas.shape[1]}x{self.canvas.shape[0]}")
        else:
            print(f"[Board.rotate] No canvas expansion needed for {rotation_deg}° rotation")

        # Rotate each card around the board's center
        for card in self.cards:
            card.rotate(rotation_deg, center=self.center_position)

        # Rotate the canvas
        h, w = self.canvas.shape[:2]
        rot_matrix = cv2.getRotationMatrix2D((self.center_position.x, self.center_position.y), rotation_deg, 1.0)
        self.canvas = cv2.warpAffine(
            self.canvas,
            rot_matrix,
            (w, h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )

    def apply_perspective(self, strength: float = 0.5, seed: int = None):
        """
        Apply perspective transformation to the board.
        Adds padding if perspective would move content outside canvas.

        Args:
            strength: 0.0 to 1.0, controls intensity of perspective effect
            seed: Random seed for reproducibility

        Returns:
            Transformation matrix for updating bounding boxes
        """
        height, width = self.canvas.shape[:2]

        # Create perspective transform
        perspective = PerspectiveTransform.from_strength(strength, width, height, seed)
        matrix = perspective.to_matrix(width, height)

        # Calculate how much the corners will move
        corners = np.array([
            [0, 0, 1],
            [width, 0, 1],
            [width, height, 1],
            [0, height, 1]
        ], dtype=np.float32).T

        transformed_corners = matrix @ corners
        transformed_corners /= transformed_corners[2, :]

        x_coords = transformed_corners[0, :]
        y_coords = transformed_corners[1, :]

        min_x = int(np.floor(np.min(x_coords)))
        max_x = int(np.ceil(np.max(x_coords)))
        min_y = int(np.floor(np.min(y_coords)))
        max_y = int(np.ceil(np.max(y_coords)))

        # Calculate needed padding
        pad_left = max(0, -min_x)
        pad_right = max(0, max_x - width)
        pad_top = max(0, -min_y)
        pad_bottom = max(0, max_y - height)

        # Add padding if needed
        if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
            print(f"[Board.apply_perspective] Expanding canvas for perspective (strength={strength}):")
            print(f"  Original size: {width}x{height}")
            print(f"  Corner movement: x=[{min_x}, {max_x}], y=[{min_y}, {max_y}]")
            print(f"  Adding padding: left={pad_left}px, right={pad_right}px, top={pad_top}px, bottom={pad_bottom}px")

            self._add_padding_asymmetric(pad_left, pad_right, pad_top, pad_bottom)

            print(f"  New canvas size: {self.canvas.shape[1]}x{self.canvas.shape[0]}")

            # Update matrix to account for padding offset
            translation_matrix = np.array([
                [1, 0, pad_left],
                [0, 1, pad_top],
                [0, 0, 1]
            ], dtype=np.float32)
            matrix = translation_matrix @ matrix
        else:
            print(f"[Board.apply_perspective] No canvas expansion needed for perspective (strength={strength})")

        # Apply perspective to canvas
        new_height, new_width = self.canvas.shape[:2]
        self.canvas = cv2.warpPerspective(
            self.canvas,
            matrix,
            (new_width, new_height),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )

        # Update all bounding boxes
        for card in self.cards:
            card.transform_card_bounding_boxes(matrix)

        return matrix

    def _add_padding(self, pad_w: int, pad_h: int):
        """
        Add symmetric padding to all sides of the canvas and update card positions.

        Args:
            pad_w: Padding to add on left and right
            pad_h: Padding to add on top and bottom
        """
        if pad_w == 0 and pad_h == 0:
            return

        h, w = self.canvas.shape[:2]
        new_canvas = np.zeros((h + 2 * pad_h, w + 2 * pad_w, 4), dtype=np.uint8)
        new_canvas[pad_h:pad_h + h, pad_w:pad_w + w] = self.canvas
        self.canvas = new_canvas

        # Update center position
        self.center_position = Coordinate(
            self.center_position.x + pad_w,
            self.center_position.y + pad_h
        )

        # Update all card positions
        for card in self.cards:
            card.offset(pad_w, pad_h)

    def _add_padding_asymmetric(self, pad_left: int, pad_right: int, pad_top: int, pad_bottom: int):
        """
        Add asymmetric padding to the canvas and update card positions.

        Args:
            pad_left: Padding to add on left
            pad_right: Padding to add on right
            pad_top: Padding to add on top
            pad_bottom: Padding to add on bottom
        """
        if pad_left == 0 and pad_right == 0 and pad_top == 0 and pad_bottom == 0:
            return

        h, w = self.canvas.shape[:2]
        new_h = h + pad_top + pad_bottom
        new_w = w + pad_left + pad_right

        new_canvas = np.zeros((new_h, new_w, 4), dtype=np.uint8)
        new_canvas[pad_top:pad_top + h, pad_left:pad_left + w] = self.canvas
        self.canvas = new_canvas

        # Update center position
        self.center_position = Coordinate(
            self.center_position.x + pad_left,
            self.center_position.y + pad_top
        )

        # Update all card positions
        for card in self.cards:
            card.offset(pad_left, pad_top)

    def crop(self, padding: int = 0):
        """
        Crop the board's canvas to the used area with optional padding.
        Also updates all card positions and bounding boxes to match the new canvas coordinates.

        Args:
            padding: Padding in pixels to be added around the cropped area
        """
        # Find the bounding box of all non-transparent pixels using the alpha channel
        alpha_channel = self.canvas[:, :, 3]
        coords = cv2.findNonZero(alpha_channel)

        if coords is None:
            print("Warning: Canvas is empty, skipping crop")
            return

        x, y, w, h = cv2.boundingRect(coords)

        # Expand the bounding box by the padding amount
        x -= padding
        y -= padding
        w += 2 * padding
        h += 2 * padding

        # Ensure the bounding box is within the image boundaries
        x = max(x, 0)
        y = max(y, 0)
        w = min(w, self.canvas.shape[1] - x)
        h = min(h, self.canvas.shape[0] - y)

        # Calculate offset to translate all coordinates
        offset_x = -x
        offset_y = -y

        # Crop the canvas to the bounding box
        self.canvas = self.canvas[y:y+h, x:x+w]

        # Update center position
        self.center_position = Coordinate(
            self.center_position.x + offset_x,
            self.center_position.y + offset_y
        )

        # Update all cards
        for card in self.cards:
            # Update card center position
            card.center_position = Coordinate(
                card.center_position.x + offset_x,
                card.center_position.y + offset_y
            )

            # Update bounding boxes
            self._offset_card_bounding_boxes(card, offset_x, offset_y)

    @staticmethod
    def _offset_card_bounding_boxes(card: Card, offset_x: int, offset_y: int):
        """
        Helper method to offset all bounding boxes of a card.

        Args:
            card: Card whose bounding boxes to offset
            offset_x: Horizontal offset
            offset_y: Vertical offset
        """
        offset = Coordinate(offset_x, offset_y)
        offset_boxes = defaultdict(list)

        for quarter_location, boxes in card.bounding_boxes.items():
            for bbox in boxes:
                offset_boxes[quarter_location].append(bbox + offset)

        card.bounding_boxes = dict(offset_boxes)

# Example usage
if __name__ == "__main__":
    """
    Example demonstrating how to create a board with 3 cards and visualize them.
    """
    import sys

    # Configuration
    CARDS_DIR = "pirate_cards"
    JSON_PATH = "pirate_cards/pirate_20_25.json"
    CARD_PREFIX = "pirate_card-"

    # Check if files exist
    if not os.path.exists(JSON_PATH):
        print(f"Error: JSON file not found at {JSON_PATH}")
        sys.exit(1)

    print("="*80)
    print("CARD BOARD EXAMPLE - 3 Cards with Different Transformations")
    print("="*80)

    # Create a board with enough space for 3 cards
    print("\nCreating board...")

    # Card 1: Top-left position, no rotation
    print("\n--- Card 1: Index 1 ---")
    try:
        card1 = Card.from_file(
            cards_dir=CARDS_DIR,
            card_index=1,
            json_path=JSON_PATH,
            card_prefix=CARD_PREFIX
        )

        # Position card 1 in the top-left area
        card1.offset_relative(1, 1)

        print(f"  Position: {card1.center_position}")
        print(f"  Rotation: {card1.rotation}°")
        print(f"  Quarters: {len(card1.quarters)}")

        # Print bounding boxes for card 1
        print(f"  Bounding boxes:")
        for quarter_loc, boxes in card1.bounding_boxes.items():
            print(f"    {quarter_loc.value}: {len(boxes)} symbols")
            for bbox in boxes:
                print(f"      {bbox.symbol_name:12} ({bbox.top_left.x:4}, {bbox.top_left.y:4}) -> ({bbox.bottom_right.x:4}, {bbox.bottom_right.y:4})")

        print("  ✓ Card 1 added successfully")

    except Exception as e:
        print(f"  ✗ Error loading card 1: {e}")
        sys.exit(1)

    board = Board(card_size=card1.original_size, number_of_cards=3)
    board.add_card(card1)

    # Card 2: Center position, 10° rotation (the test card - pirate_card-19.png)
    print("\n--- Card 2: Index 19 (Test Card) ---")
    try:
        card2 = Card.from_file(
            cards_dir=CARDS_DIR,
            card_index=19,
            json_path=JSON_PATH,
            card_prefix=CARD_PREFIX
        )

        # Rotate card 2
        card2.rotate(10)

        # Position card 2 in the center (no offset needed)
        print(f"  Position: {card2.center_position}")
        print(f"  Rotation: {card2.rotation}°")
        print(f"  Bounding boxes:")
        for quarter_loc, boxes in card2.bounding_boxes.items():
            print(f"    {quarter_loc.value}: {len(boxes)} symbols")
            for bbox in boxes:
                print(f"      {bbox.symbol_name:12} ({bbox.top_left.x:4}, {bbox.top_left.y:4}) -> ({bbox.bottom_right.x:4}, {bbox.bottom_right.y:4})")

        board.add_card(card2, QuarterLocation.TOP_LEFT, to_card_index=0, to_quarter=QuarterLocation.BOTTOM_RIGHT)
        print("  ✓ Card 2 added successfully")

    except Exception as e:
        print(f"  ✗ Error loading card 2: {e}")
        sys.exit(1)

    # Card 3: Bottom-right position, -10° rotation
    print("\n--- Card 3: Index 25 ---")
    try:
        card3 = Card.from_file(
            cards_dir=CARDS_DIR,
            card_index=25,
            json_path=JSON_PATH,
            card_prefix=CARD_PREFIX
        )

        # Rotate card 3
        card3.rotate(110)

        print(f"  Position: {card3.center_position}")
        print(f"  Rotation: {card3.rotation}°")
        print(f"  Quarters: {len(card3.quarters)}")

        # Print bounding boxes for card 3
        print(f"  Bounding boxes:")
        for quarter_loc, boxes in card3.bounding_boxes.items():
            print(f"    {quarter_loc.value}: {len(boxes)} symbols")
            for bbox in boxes:
                print(f"      {bbox.symbol_name:12} ({bbox.top_left.x:4}, {bbox.top_left.y:4}) -> ({bbox.bottom_right.x:4}, {bbox.bottom_right.y:4})")

        board.add_card(card3, QuarterLocation.TOP_LEFT, to_card_index=1, to_quarter=QuarterLocation.BOTTOM_RIGHT)
        print("  ✓ Card 3 added successfully")

    except Exception as e:
        print(f"  ✗ Error loading card 3: {e}")
        sys.exit(1)

    # Render the board
    print("\n" + "="*80)
    print("Rendering board with all cards...")

    # Apply geometric transformations (these affect bounding boxes)
    print("\nApplying geometric transformations...")
    print("  - Rotating board by 30 degrees")
    board.rotate(30)
    print("  - Applying perspective distortion (strength: 2)")
    board.apply_perspective(2)

    # Crop the board to used area
    print("\n" + "="*80)
    print("Cropping board to used area...")
    print(f"Original canvas size: {board.canvas.shape[1]}x{board.canvas.shape[0]}")

    board.crop(padding=100)

    print(f"Cropped canvas size: {board.canvas.shape[1]}x{board.canvas.shape[0]}")
    print("✓ Board cropped successfully")

    # Draw bounding boxes on the canvas
    canvas_with_boxes = cv2.cvtColor(board.canvas.copy(), cv2.COLOR_BGRA2BGR)

    for card_idx, card in enumerate(board.cards):
        print(f"\nDrawing bounding boxes for Card {card_idx + 1}...")
        for quarter_loc, boxes in card.bounding_boxes.items():
            # Find the corresponding quarter object to check visibility
            quarter = next((q for q in card.quarters if q.location == quarter_loc), None)

            # Skip drawing if quarter is not visible
            if quarter is None or not quarter.visible:
                print(f"  Skipping {quarter_loc.value} (not visible)")
                continue

            for bbox in boxes:
                # Choose color based on symbol type
                is_quarter = bbox.symbol_name == "quarter"
                color = (0, 0, 255) if is_quarter else (0, 255, 0)  # Red for quarters, Green for symbols

                # Draw bounding box
                cv2.rectangle(
                    canvas_with_boxes,
                    (bbox.top_left.x, bbox.top_left.y),
                    (bbox.bottom_right.x, bbox.bottom_right.y),
                    color,
                    2
                )

                # Draw center point
                center_x = (bbox.top_left.x + bbox.bottom_right.x) // 2
                center_y = (bbox.top_left.y + bbox.bottom_right.y) // 2
                cv2.circle(canvas_with_boxes, (center_x, center_y), 3, color, -1)

                # Draw symbol name
                label = bbox.symbol_name
                # Position label above the box
                label_x = bbox.top_left.x
                label_y = bbox.top_left.y - 5

                # If label would go off top of image, put it below instead
                if label_y < 15:
                    label_y = bbox.bottom_right.y + 15

                cv2.putText(
                    canvas_with_boxes,
                    label,
                    (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                    cv2.LINE_AA
                )

    print("✓ Board rendered successfully with bounding boxes")

    # Display the board without effects
    print("\nDisplaying board with bounding boxes (before visual effects)...")
    print("RED boxes show quarter bounding boxes")
    print("GREEN boxes show symbol bounding boxes with labels")
    print("Press any key to continue...")
    cv2.imshow("Card Board - Before Visual Effects", canvas_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Apply visual effects to the board (these don't affect bounding boxes)
    print("\n" + "="*80)
    print("Applying visual camera effects to the board...")
    print("="*80)

    # Keep canvas in BGRA format so background can be added with alpha blending
    print("\nApplying 'moderate' preset visual effects...")
    print("This includes: background, glare, blur, noise, lighting, vignette, compression")
    print("Note: Perspective and rotation were already applied geometrically")

    # Apply only visual effects (no geometric transformation)
    # Pass BGRA canvas so background can be blended properly
    canvas_with_effects = apply_camera_effects_to_composite(
        board.canvas,  # Keep as BGRA
        effect_preset="heavy",
        debug=False
    )

    print("✓ Visual effects applied successfully")

    # Draw bounding boxes on the effects image (bounding boxes are still valid)
    canvas_with_effects_and_boxes = canvas_with_effects.copy()

    for card_idx, card in enumerate(board.cards):
        print(f"\nDrawing bounding boxes on effects image for Card {card_idx + 1}...")
        for quarter_loc, boxes in card.bounding_boxes.items():
            # Find the corresponding quarter object to check visibility
            quarter = next((q for q in card.quarters if q.location == quarter_loc), None)

            # Skip drawing if quarter is not visible
            if quarter is None or not quarter.visible:
                continue

            for bbox in boxes:
                # Choose color based on symbol type
                is_quarter = bbox.symbol_name == "quarter"
                color = (0, 0, 255) if is_quarter else (0, 255, 0)  # Red for quarters, Green for symbols

                # Draw bounding box
                cv2.rectangle(
                    canvas_with_effects_and_boxes,
                    (bbox.top_left.x, bbox.top_left.y),
                    (bbox.bottom_right.x, bbox.bottom_right.y),
                    color,
                    2
                )

                # Draw center point
                center_x = (bbox.top_left.x + bbox.bottom_right.x) // 2
                center_y = (bbox.top_left.y + bbox.bottom_right.y) // 2
                cv2.circle(canvas_with_effects_and_boxes, (center_x, center_y), 3, color, -1)

                # Draw symbol name
                label = bbox.symbol_name
                label_x = bbox.top_left.x
                label_y = bbox.top_left.y - 5

                if label_y < 15:
                    label_y = bbox.bottom_right.y + 15

                cv2.putText(
                    canvas_with_effects_and_boxes,
                    label,
                    (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                    cv2.LINE_AA
                )

    # Display board with visual effects and bounding boxes
    print("\nDisplaying board with visual effects and bounding boxes...")
    print("RED boxes show quarter bounding boxes")
    print("GREEN boxes show symbol bounding boxes")
    print("Press any key to close...")
    cv2.imshow("Card Board - With Visual Effects + Bounding Boxes", canvas_with_effects_and_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save all outputs
    output_path_before = "output_3cards_board_before_effects.png"
    output_path_boxes = "output_3cards_board_with_boxes.png"
    output_path_effects = "output_3cards_board_with_effects.png"
    output_path_effects_boxes = "output_3cards_board_with_effects_and_boxes.png"

    print(f"\nSaving outputs...")
    cv2.imwrite(output_path_before, cv2.cvtColor(board.canvas, cv2.COLOR_BGRA2BGR))
    cv2.imwrite(output_path_boxes, canvas_with_boxes)
    cv2.imwrite(output_path_effects, canvas_with_effects)
    cv2.imwrite(output_path_effects_boxes, canvas_with_effects_and_boxes)

    print("\n" + "="*80)
    print("Example completed successfully!")
    print(f"Output files:")
    print(f"  - {output_path_before} (board after geometric transforms)")
    print(f"  - {output_path_boxes} (with bounding boxes, before visual effects)")
    print(f"  - {output_path_effects} (with visual effects)")
    print(f"  - {output_path_effects_boxes} (with visual effects + bounding boxes)")
    print("="*80)
