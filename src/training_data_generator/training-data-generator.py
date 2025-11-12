import json
import random
import os
from typing import List, Dict, Tuple
import cv2
import numpy as np

from new_image_generator import (
    Board, Card, QuarterLocation, BoundingBox, Coordinate
)
from image_effects import apply_camera_effects_to_composite


class TrainingDataGenerator:
    """Generate training images with cards and bounding box annotations."""

    def __init__(
            self,
            cards_dir: str,
            json_path: str,
            output_dir: str = "training_data",
            card_prefix: str = "pirate_card-",
            seed: int = None,
            save_intermediate: bool = True,
            card_offset_percent: float = 0.1,
            show_hidden_symbols: bool = False,
            save_bounding_boxes: bool = True
    ):
        """
        Initialize the training data generator.

        Args:
            cards_dir: Directory containing card images
            json_path: Path to card definitions JSON
            output_dir: Directory to save generated training data
            card_prefix: Prefix for card image files
            seed: Random seed for reproducibility
            save_intermediate: Whether to save intermediate images after each operation
            card_offset_percent: Random offset range as percentage of card size (0.0-1.0)
            show_hidden_symbols: Whether to show hidden symbols in visualizations (default: False)
        """
        self.cards_dir = cards_dir
        self.json_path = json_path
        self.output_dir = output_dir
        self.card_prefix = card_prefix
        self.save_intermediate = save_intermediate
        self.card_offset_percent = card_offset_percent
        self.show_hidden_symbols = show_hidden_symbols
        self.save_bounding_boxes = save_bounding_boxes

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)

        # Create intermediate images directory if needed
        if self.save_intermediate:
            os.makedirs(os.path.join(output_dir, "intermediate"), exist_ok=True)

    def generate_training_sample(
            self,
            sample_id: int,
            num_cards: int = 6,
            max_card_rotation: float = 45.0,
            max_board_rotation: float = 30.0,
            perspective_strength: float = 0.5,
            effect_preset: str = "moderate",
            available_card_indices: List[int] = None
    ) -> Dict:
        """
        Generate a single training sample with random card placement and transformations.

        Args:
            sample_id: Unique identifier for this sample
            num_cards: Number of cards to place on the board
            max_card_rotation: Maximum rotation in degrees for individual cards
            max_board_rotation: Maximum rotation in degrees for the entire board
            perspective_strength: Strength of perspective distortion (0.0-1.0)
            effect_preset: Camera effect preset ("light", "moderate", "heavy")
            available_card_indices: List of card indices to choose from (if None, uses 1-25)

        Returns:
            Dictionary containing annotation data in neural network training format
        """
        print(f"\n{'=' * 80}")
        print(f"Generating Training Sample #{sample_id}")
        print(f"{'=' * 80}")

        if available_card_indices is None:
            available_card_indices = list(range(1, 26))  # Cards 1-25

        # Randomly select cards
        selected_indices = random.sample(available_card_indices, min(num_cards, len(available_card_indices)))
        print(f"Selected cards: {selected_indices}")

        # Load first card to get size
        first_card = Card.from_file(
            cards_dir=self.cards_dir,
            card_index=selected_indices[0],
            json_path=self.json_path,
            card_prefix=self.card_prefix
        )

        # Apply random 90-degree rotation to first card
        first_card_90deg_rotation = random.choice([0, 90, 180, 270])
        if first_card_90deg_rotation > 0:
            first_card.rotate(first_card_90deg_rotation)
            print(f"\nCard 1: Index {selected_indices[0]} (initial card)")
            print(f"  90° rotation: {first_card_90deg_rotation}°")

        # Create board
        board = Board(card_size=first_card.original_size, number_of_cards=num_cards)

        # Offset first card to center of canvas
        canvas_center_x = board.canvas.shape[1] // 2
        canvas_center_y = board.canvas.shape[0] // 2
        first_card_offset_x = canvas_center_x - first_card.center_position.x
        first_card_offset_y = canvas_center_y - first_card.center_position.y
        first_card.offset(first_card_offset_x, first_card_offset_y)
        print(f"  Offset to canvas center: ({first_card_offset_x}, {first_card_offset_y})")

        # Add first card
        board.add_card(first_card)

        # Save intermediate image after first card
        self._save_intermediate_image(board, sample_id, "01_card_1_added")

        # Add remaining cards with random placement and rotation
        for i, card_idx in enumerate(selected_indices[1:], start=2):
            print(f"\nCard {i}: Index {card_idx}")

            # Load card
            card = Card.from_file(
                cards_dir=self.cards_dir,
                card_index=card_idx,
                json_path=self.json_path,
                card_prefix=self.card_prefix
            )

            # Apply random 90-degree rotation first
            card_90deg_rotation = random.choice([0, 90, 180, 270])
            if card_90deg_rotation > 0:
                card.rotate(card_90deg_rotation)
                print(f"  90° rotation: {card_90deg_rotation}°")

            # Then apply random continuous rotation
            card_rotation = random.uniform(-max_card_rotation, max_card_rotation)
            card.rotate(int(card_rotation))
            print(f"  Fine rotation: {card_rotation:.1f}°")
            print(f"  Total rotation: {card_90deg_rotation + card_rotation:.1f}°")

            # Random placement on top of a random quarter of a random previous card
            target_card_idx = random.randint(0, len(board.cards) - 1)
            target_quarter = random.choice(list(QuarterLocation))
            source_quarter = random.choice(list(QuarterLocation))

            print(f"  Placing {source_quarter.value} on card {target_card_idx + 1}'s {target_quarter.value}")

            # Apply random offset proportional to card size
            if self.card_offset_percent > 0:
                max_offset = int(card.original_size * self.card_offset_percent)
                offset_x = random.randint(-max_offset, max_offset)
                offset_y = random.randint(-max_offset, max_offset)
                card.offset(offset_x, offset_y)
                print(f"  Random offset: ({offset_x}, {offset_y}) px (±{self.card_offset_percent*100:.0f}% of card size)")

            board.add_card(
                card,
                quarter_location=source_quarter,
                to_card_index=target_card_idx,
                to_quarter=target_quarter,
                on_top=True
            )
            # Save intermediate image after each card
            self._save_intermediate_image(board, sample_id, f"{i:02d}_card_{i}_added")

        # Apply board transformations
        print(f"\n{'=' * 80}")
        print("Applying Board Transformations")
        print(f"{'=' * 80}")

        # Random board rotation
        board_rotation = random.uniform(-max_board_rotation, max_board_rotation)
        print(f"Board rotation: {board_rotation:.1f}°")
        board.rotate(int(board_rotation))

        # Save intermediate image after board rotation
        self._save_intermediate_image(board, sample_id, f"{num_cards+1:02d}_board_rotated")

        # Apply perspective
        print(f"Perspective strength: {perspective_strength}")
        board.apply_perspective(strength=perspective_strength, seed=random.randint(0, 100000))

        # Save intermediate image after perspective
        self._save_intermediate_image(board, sample_id, f"{num_cards+2:02d}_perspective_applied")

        # Crop to used area
        print("Cropping to used area...")
        board.crop(padding=150)

        # Save intermediate image after cropping
        self._save_intermediate_image(board, sample_id, f"{num_cards+3:02d}_cropped")

        # Get image dimensions before effects
        height, width = board.canvas.shape[:2]
        print(f"Final board size: {width}x{height}")

        # Collect annotations before visual effects
        annotations = self._collect_annotations(board, sample_id, width, height)

        # Apply visual effects
        print(f"\nApplying visual effects (preset: {effect_preset})...")
        canvas_with_effects = apply_camera_effects_to_composite(
            board.canvas,
            effect_preset=effect_preset,
            debug=False
        )

        # Save intermediate image after visual effects
        if self.save_intermediate:
            intermediate_path = os.path.join(
                self.output_dir,
                "intermediate",
                f"sample_{sample_id:05d}_{num_cards+4:02d}_visual_effects.png"
            )
            cv2.imwrite(intermediate_path, canvas_with_effects)
            print(f"  Saved intermediate: {intermediate_path}")

        # Save outputs
        self._save_outputs(
            sample_id=sample_id,
            image=canvas_with_effects,
            annotations=annotations,
            board=board
        )

        print(f"\n{'=' * 80}")
        print(f"Sample #{sample_id} completed successfully")
        print(f"{'=' * 80}\n")

        return annotations

    def _collect_annotations(
            self,
            board: Board,
            sample_id: int,
            image_width: int,
            image_height: int
    ) -> Dict:
        """
        Collect bounding box annotations in a format suitable for neural network training.

        Format follows COCO-style structure:
        {
            "image": {
                "id": int,
                "width": int,
                "height": int,
                "file_name": str
            },
            "annotations": [
                {
                    "id": int,
                    "image_id": int,
                    "category_id": int,
                    "category_name": str,
                    "bbox": [x, y, width, height],  # Top-left corner + dimensions
                    "bbox_normalized": [x, y, width, height],  # Normalized to [0, 1]
                    "area": float,
                    "card_id": int,
                    "quarter_location": str,
                    "visible": bool
                }
            ],
            "categories": [
                {
                    "id": int,
                    "name": str,
                    "supercategory": str
                }
            ]
        }

        Args:
            board: Board object containing cards and bounding boxes
            sample_id: Unique identifier for this sample
            image_width: Width of the final image
            image_height: Height of the final image

        Returns:
            Dictionary containing annotation data
        """
        annotations_list = []
        categories_set = set()
        annotation_id = 0

        # Collect all bounding boxes from all cards
        for card_idx, card in enumerate(board.cards):
            for quarter_loc, boxes in card.bounding_boxes.items():
                # Find the corresponding quarter object to check visibility
                quarter = next((q for q in card.quarters if q.location == quarter_loc), None)
                is_visible = quarter is not None and quarter.visible

                for bbox in boxes:
                    # Add category to set
                    categories_set.add(bbox.symbol_name)

                    # Calculate bbox in COCO format: [x, y, width, height]
                    x = bbox.top_left.x
                    y = bbox.top_left.y
                    w = bbox.bottom_right.x - bbox.top_left.x
                    h = bbox.bottom_right.y - bbox.top_left.y

                    # Normalized coordinates (0 to 1)
                    x_norm = x / image_width
                    y_norm = y / image_height
                    w_norm = w / image_width
                    h_norm = h / image_height

                    # Area
                    area = w * h

                    annotation = {
                        "id": annotation_id,
                        "image_id": sample_id,
                        "category_id": None,  # Will be filled after creating category mapping
                        "category_name": bbox.symbol_name,
                        "bbox": [int(x), int(y), int(w), int(h)],
                        "bbox_normalized": [
                            float(x_norm),
                            float(y_norm),
                            float(w_norm),
                            float(h_norm)
                        ],
                        "area": float(area),
                        "card_id": card_idx,
                        "quarter_location": quarter_loc.value,
                        "visible": is_visible
                    }

                    annotations_list.append(annotation)
                    annotation_id += 1

        # Create category mapping
        categories = []
        category_name_to_id = {}
        for cat_id, cat_name in enumerate(sorted(categories_set)):
            categories.append({
                "id": cat_id,
                "name": cat_name,
                "supercategory": "symbol"
            })
            category_name_to_id[cat_name] = cat_id

        # Fill in category_id for annotations
        for annotation in annotations_list:
            annotation["category_id"] = category_name_to_id[annotation["category_name"]]

        # Construct final annotation structure
        result = {
            "image": {
                "id": sample_id,
                "width": image_width,
                "height": image_height,
                "file_name": f"sample_{sample_id:05d}.png"
            },
            "annotations": annotations_list,
            "categories": categories
        }

        return result

    def _save_outputs(
            self,
            sample_id: int,
            image: np.ndarray,
            annotations: Dict,
            board: Board
    ):
        """
        Save generated image, annotations, and visualization.

        Args:
            sample_id: Unique identifier for this sample
            image: Final image with effects applied
            annotations: Annotation dictionary
            board: Board object for visualization
        """
        # File paths
        image_path = os.path.join(
            self.output_dir,
            "images",
            f"sample_{sample_id:05d}.png"
        )
        annotation_path = os.path.join(
            self.output_dir,
            "annotations",
            f"sample_{sample_id:05d}.json"
        )
        visualization_path = os.path.join(
            self.output_dir,
            "visualizations",
            f"sample_{sample_id:05d}_annotated.png"
        )

        # Save image
        cv2.imwrite(image_path, image)
        print(f"Saved image: {image_path}")

        # Save annotations
        with open(annotation_path, 'w') as f:
            json.dump(annotations, f, indent=2)
        print(f"Saved annotations: {annotation_path}")

        if self.save_bounding_boxes:
            # Create and save visualization
            visualization = self._create_visualization(image.copy(), annotations)
            cv2.imwrite(visualization_path, visualization)
            print(f"Saved visualization: {visualization_path}")

    def _create_visualization(
            self,
            image: np.ndarray,
            annotations: Dict
    ) -> np.ndarray:
        """
        Draw bounding boxes on the image for visualization.

        Args:
            image: Image to draw on
            annotations: Annotation dictionary

        Returns:
            Image with bounding boxes drawn
        """
        # Define colors for different visibility states
        COLOR_VISIBLE = (0, 255, 0)  # Green
        COLOR_HIDDEN = (128, 128, 128)  # Gray

        for ann in annotations["annotations"]:
            is_visible = ann["visible"]

            # Skip hidden symbols if show_hidden_symbols is False
            if not is_visible and not self.show_hidden_symbols:
                continue

            x, y, w, h = ann["bbox"]

            # Choose color based on visibility
            color = COLOR_VISIBLE if is_visible else COLOR_HIDDEN
            thickness = 2 if is_visible else 1

            # Draw bounding box
            cv2.rectangle(
                image,
                (x, y),
                (x + w, y + h),
                color,
                thickness
            )

            # Draw center point
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.circle(image, (center_x, center_y), 3, color, -1)

            # Draw label
            label = f"{ann['category_name']}"
            if not is_visible:
                label += " (hidden)"

            # Position label above the box
            label_x = x
            label_y = y - 5

            # If label would go off top of image, put it below instead
            if label_y < 15:
                label_y = y + h + 15

            # Draw label background
            (label_w, label_h), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                1
            )
            cv2.rectangle(
                image,
                (label_x, label_y - label_h - 2),
                (label_x + label_w, label_y + 2),
                color,
                -1
            )

            # Draw label text
            cv2.putText(
                image,
                label,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),  # Black text
                1,
                cv2.LINE_AA
            )

        return image

    def generate_dataset(
            self,
            num_samples: int,
            num_cards: int = 6,
            max_card_rotation: float = 45.0,
            max_board_rotation: float = 30.0,
            perspective_strength_range: Tuple[float, float] = (0.3, 0.7),
            effect_presets: List[str] = None,
            available_card_indices: List[int] = None
    ):
        """
        Generate a complete dataset with multiple samples.

        Args:
            num_samples: Number of training samples to generate
            num_cards: Number of cards per sample
            max_card_rotation: Maximum rotation in degrees for individual cards
            max_board_rotation: Maximum rotation in degrees for the entire board
            perspective_strength_range: Range of perspective strength (min, max)
            effect_presets: List of effect presets to randomly choose from
            available_card_indices: List of card indices to choose from
        """
        if effect_presets is None:
            effect_presets = ["light", "moderate", "heavy"]

        print(f"\n{'=' * 80}")
        print(f"GENERATING DATASET: {num_samples} samples")
        print(f"{'=' * 80}")
        print(f"Configuration:")
        print(f"  Cards per sample: {num_cards}")
        print(f"  Max card rotation: ±{max_card_rotation}°")
        print(f"  Max board rotation: ±{max_board_rotation}°")
        print(f"  Perspective strength: {perspective_strength_range[0]}-{perspective_strength_range[1]}")
        print(f"  Effect presets: {effect_presets}")
        print(f"{'=' * 80}\n")

        # Create dataset-level annotation file (COCO format)
        dataset_annotations = {
            "info": {
                "description": "Card Symbol Detection Dataset",
                "version": "1.0",
                "year": 2024
            },
            "images": [],
            "annotations": [],
            "categories": []
        }

        all_categories = set()

        for sample_id in range(num_samples):
            # Random parameters for this sample
            perspective_strength = random.uniform(*perspective_strength_range)
            effect_preset = random.choice(effect_presets)

            # Generate sample
            annotations = self.generate_training_sample(
                sample_id=sample_id,
                num_cards=num_cards,
                max_card_rotation=max_card_rotation,
                max_board_rotation=max_board_rotation,
                perspective_strength=perspective_strength,
                effect_preset=effect_preset,
                available_card_indices=available_card_indices
            )

            # Add to dataset annotations
            dataset_annotations["images"].append(annotations["image"])
            dataset_annotations["annotations"].extend(annotations["annotations"])

            # Collect categories
            for cat in annotations["categories"]:
                all_categories.add(cat["name"])

        # Create final category list
        dataset_annotations["categories"] = [
            {"id": cat_id, "name": cat_name, "supercategory": "symbol"}
            for cat_id, cat_name in enumerate(sorted(all_categories))
        ]

        # Save dataset-level annotations
        dataset_path = os.path.join(self.output_dir, "dataset_annotations.json")
        with open(dataset_path, 'w') as f:
            json.dump(dataset_annotations, f, indent=2)

        print(f"\n{'=' * 80}")
        print(f"DATASET GENERATION COMPLETE")
        print(f"{'=' * 80}")
        print(f"Generated {num_samples} samples")
        print(f"Total annotations: {len(dataset_annotations['annotations'])}")
        print(f"Total categories: {len(dataset_annotations['categories'])}")
        print(f"Dataset annotations saved to: {dataset_path}")
        print(f"{'=' * 80}\n")


    def _save_intermediate_image(self, board: Board, sample_id: int, step_name: str):
        """
        Save an intermediate image of the board state with bounding boxes.

        Args:
            board: Current board state
            sample_id: Sample identifier
            step_name: Name of the current step (for filename)
        """
        if not self.save_intermediate:
            return

        # Convert canvas to BGR for saving
        canvas_bgr = cv2.cvtColor(board.canvas, cv2.COLOR_BGRA2BGR)

        # Draw bounding boxes on the intermediate image
        for card_idx, card in enumerate(board.cards):
            for quarter_loc, boxes in card.bounding_boxes.items():
                # Find the corresponding quarter object to check visibility
                quarter = next((q for q in card.quarters if q.location == quarter_loc), None)
                is_visible = quarter is not None and quarter.visible

                # Skip hidden symbols if show_hidden_symbols is False
                if not is_visible and not self.show_hidden_symbols:
                    continue

                # Choose color based on visibility
                color = (0, 255, 0) if is_visible else (128, 128, 128)  # Green or Gray
                thickness = 2 if is_visible else 1

                for bbox in boxes:
                    # Draw bounding box
                    cv2.rectangle(
                        canvas_bgr,
                        (bbox.top_left.x, bbox.top_left.y),
                        (bbox.bottom_right.x, bbox.bottom_right.y),
                        color,
                        thickness
                    )

                    # Draw center point
                    center_x = (bbox.top_left.x + bbox.bottom_right.x) // 2
                    center_y = (bbox.top_left.y + bbox.bottom_right.y) // 2
                    cv2.circle(canvas_bgr, (center_x, center_y), 3, color, -1)

                    # Draw symbol name label (always show for visible, conditionally for hidden)
                    if is_visible or self.show_hidden_symbols:
                        label = bbox.symbol_name
                        if not is_visible:
                            label += " (hidden)"

                        label_x = bbox.top_left.x
                        label_y = bbox.top_left.y - 5

                        # If label would go off top, put it below
                        if label_y < 15:
                            label_y = bbox.bottom_right.y + 15

                        # Draw label background
                        (label_w, label_h), _ = cv2.getTextSize(
                            label,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            1
                        )
                        cv2.rectangle(
                            canvas_bgr,
                            (label_x, label_y - label_h - 2),
                            (label_x + label_w, label_y + 2),
                            color,
                            -1
                        )

                        # Draw label text
                        cv2.putText(
                            canvas_bgr,
                            label,
                            (label_x, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (0, 0, 0),  # Black text
                            1,
                            cv2.LINE_AA
                        )

        # Add step name annotation in top-left corner
        step_label = f"Step: {step_name}"
        cv2.rectangle(canvas_bgr, (10, 10), (300, 50), (255, 255, 255), -1)
        cv2.rectangle(canvas_bgr, (10, 10), (300, 50), (0, 0, 0), 2)
        cv2.putText(
            canvas_bgr,
            step_label,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )

        # Save the image
        intermediate_path = os.path.join(
            self.output_dir,
            "intermediate",
            f"sample_{sample_id:05d}_{step_name}.png"
        )
        cv2.imwrite(intermediate_path, canvas_bgr)
        print(f"  Saved intermediate with bboxes: {intermediate_path}")


if __name__ == "__main__":
    """
    Example: Generate a training dataset with configurable card offsets.
    """

    # Configuration
    CARDS_DIR = "../../pirate_cards"
    JSON_PATH = "../../pirate_cards/pirate_20_25.json"
    OUTPUT_DIR = "training_data"

    # Create generator with configurable offset
    generator = TrainingDataGenerator(
        cards_dir=CARDS_DIR,
        json_path=JSON_PATH,
        output_dir=OUTPUT_DIR,
        seed=121,  # For reproducibility
        save_intermediate=True,  # Enable intermediate image saving
        card_offset_percent=0.15,  # 15% of card size for random offset
        show_hidden_symbols=False,  # Don't show hidden symbols in visualizations
        save_bounding_boxes=True
    )

    # Generate dataset
    generator.generate_dataset(
        num_samples=1,
        num_cards=6,
        max_card_rotation=15.0,
        max_board_rotation=15.0,
        perspective_strength_range=(0.3, 0.7),
        effect_presets=["light", "moderate", "heavy"],
        available_card_indices=list(range(1, 120))  # Use all 25 cards
    )

    print("\nDataset generation completed!")
    print(f"Check the '{OUTPUT_DIR}' directory for:")
    print("  - images/: Training images")
    print("  - annotations/: Individual JSON annotations")
    print("  - visualizations/: Images with bounding boxes drawn")
    print("  - intermediate/: Step-by-step images showing each operation")
    print("  - dataset_annotations.json: Complete dataset in COCO format")

