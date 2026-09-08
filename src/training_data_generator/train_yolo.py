"""
Train YOLO model on custom card game dataset.
This script converts custom JSON annotations to YOLO format and trains a YOLOv8 model.
"""

import json
import os
from pathlib import Path
import shutil
import yaml
from sympy.printing.pytorch import torch
from ultralytics import YOLO
from PIL import Image


class YOLOTrainer:
    def __init__(self, data_dir: str, output_dir: str = "yolo_dataset"):
        """
        Initialize YOLO trainer.

        Args:
            data_dir: Directory containing images and annotations
            output_dir: Directory where YOLO dataset will be created
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.images_dir = self.data_dir / "images"
        self.annotations_dir = self.data_dir / "annotations"

        # Create YOLO dataset structure
        self.yolo_train_images = self.output_dir / "images" / "train"
        self.yolo_train_labels = self.output_dir / "labels" / "train"
        self.yolo_val_images = self.output_dir / "images" / "val"
        self.yolo_val_labels = self.output_dir / "labels" / "val"

        for path in [self.yolo_train_images, self.yolo_train_labels,
                     self.yolo_val_images, self.yolo_val_labels]:
            path.mkdir(parents=True, exist_ok=True)

    def convert_bbox_to_yolo(self, bbox: list, img_width: int, img_height: int) -> list:
        """
        Convert bbox from [x, y, width, height] to YOLO format [x_center, y_center, width, height] (normalized).

        Args:
            bbox: Bounding box in [x, y, width, height] format
            img_width: Image width
            img_height: Image height

        Returns:
            YOLO format bbox [x_center, y_center, width, height] (all normalized 0-1)
        """
        x, y, w, h = bbox
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        width = w / img_width
        height = h / img_height
        return [x_center, y_center, width, height]

    def load_categories(self, json_file: str) -> dict:
        """Load categories from JSON file."""
        with open(json_file, 'r') as f:
            data = json.load(f)

        categories = {}
        for cat in data['categories']:
            categories[cat['id']] = cat['name']

        return categories

    def convert_annotation(self, json_file: str, is_train: bool = True):
        """
        Convert single JSON annotation to YOLO format.

        Args:
            json_file: Path to JSON annotation file
            is_train: Whether this is training data (vs validation)
        """
        with open(json_file, 'r') as f:
            data = json.load(f)

        image_info = data['image']
        annotations = data['annotations']

        # Determine output paths
        if is_train:
            label_dir = self.yolo_train_labels
            image_dir = self.yolo_train_images
        else:
            label_dir = self.yolo_val_labels
            image_dir = self.yolo_val_images

        # Create YOLO format label file
        label_file = label_dir / f"{Path(image_info['file_name']).stem}.txt"

        with open(label_file, 'w') as f:
            for ann in annotations:
                # Skip invisible annotations if desired
                # if not ann.get('visible', True):
                #     continue

                category_id = ann['category_id']
                bbox = ann['bbox']

                # Convert to YOLO format
                yolo_bbox = self.convert_bbox_to_yolo(
                    bbox,
                    image_info['width'],
                    image_info['height']
                )

                # Write: class_id x_center y_center width height
                f.write(f"{category_id} {' '.join(map(str, yolo_bbox))}\n")

        # Copy image to YOLO dataset
        src_image = self.images_dir / image_info['file_name']
        dst_image = image_dir / image_info['file_name']

        if src_image.exists():
            shutil.copy2(src_image, dst_image)
        else:
            print(f"Warning: Image not found: {src_image}")

    def prepare_dataset(self, train_split: float = 0.8):
        """
        Convert all annotations to YOLO format and split into train/val.

        Args:
            train_split: Fraction of data to use for training
        """
        # Get all annotation files
        annotation_files = sorted(self.annotations_dir.glob("*.json"))

        if not annotation_files:
            raise ValueError(f"No JSON files found in {self.annotations_dir}")

        # Load categories from first file
        categories = self.load_categories(annotation_files[0])

        # Split into train/val
        split_idx = int(len(annotation_files) * train_split)
        train_files = annotation_files[:split_idx]
        val_files = annotation_files[split_idx:]

        print(f"Converting {len(train_files)} training files...")
        for json_file in train_files:
            self.convert_annotation(json_file, is_train=True)

        print(f"Converting {len(val_files)} validation files...")
        for json_file in val_files:
            self.convert_annotation(json_file, is_train=False)

        # Create data.yaml
        self.create_data_yaml(categories)

        print(f"Dataset prepared at {self.output_dir}")
        print(f"Training samples: {len(train_files)}")
        print(f"Validation samples: {len(val_files)}")

    def create_data_yaml(self, categories: dict):
        """Create YOLO data.yaml configuration file."""
        data_yaml = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(categories),
            'names': [categories[i] for i in sorted(categories.keys())]
        }

        yaml_path = self.output_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

        print(f"Created data.yaml at {yaml_path}")

    def train(self, model_size='n', epochs=100, img_size=640, batch_size=16, device='cpu', resume=False, patience=10,
              name='card_game_yolo'):
        """Train YOLO model with option to resume."""
        # Check if we should resume from checkpoint
        checkpoint_path = Path('runs/detect/card_game_yolo/weights/last.pt')

        if resume and checkpoint_path.exists():
            print(f"Resuming training from {checkpoint_path}")
            model = YOLO(str(checkpoint_path))
        else:
            print(f"Starting new training with YOLOv8{model_size}")
            model = YOLO(f'yolov8{model_size}.pt')

        # Train
        results = model.train(
            data=str(self.output_dir / 'data.yaml'),
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            device=device,
            project='runs/detect',
            exist_ok=True,  # Allow overwriting
            save=True,
            plots=True,
            patience=patience,
            name=name
        )

        return results

    def validate(self, model_path: str):
        """Validate trained model."""
        model = YOLO(model_path)
        metrics = model.val(data=str(self.output_dir / 'data.yaml'))
        return metrics


def main():
    """Main training script."""
    DATA_DIR = "training_data_big"
    OUTPUT_DIR = "yolo_dataset"

    trainer = YOLOTrainer(DATA_DIR, OUTPUT_DIR)

    # Prepare dataset (skip if already prepared)
    if not Path(OUTPUT_DIR).exists():
        print("Preparing dataset...")
        trainer.prepare_dataset(train_split=0.8)

    # Train model with resume option
    print("\nStarting training...")
    results = trainer.train(
        model_size='s',
        epochs=200,
        img_size=640,
        batch_size=8,
        device='mps' if torch.backends.mps.is_available() else 'cpu',
        resume=True,  # Enable resuming
        patience=50,
        name='card_game_yolo_10k',
    )


if __name__ == "__main__":
    main()
# YOLO Training for Card Game Detection

