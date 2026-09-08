# YOLO Training for Card Game Detection

This directory contains scripts to train a YOLOv8 model on the card game dataset.

## Setup

1. Install requirements:
```bash
pip install -r requirements_yolo.txt
```

2. Ensure your data is organized as:
```
training_data/
├── images/
│   ├── sample_00000.png
│   ├── sample_00001.png
│   └── ...
└── annotations/
    ├── sample_00000.json
    ├── sample_00001.json
    └── ...
```

## Training

Run the training script:
```bash
python train_yolo.py
```

This will:
1. Convert your JSON annotations to YOLO format
2. Create train/val split (80/20 by default)
3. Train a YOLOv8 nano model
4. Save the best model to `runs/detect/card_game_yolo/weights/best.pt`

## Customization

Edit `train_yolo.py` to customize:
- `model_size`: 'n', 's', 'm', 'l', 'x' (nano to extra-large)
- `epochs`: Number of training epochs
- `img_size`: Input image size
- `batch_size`: Batch size
- `train_split`: Train/validation split ratio

## Inference

After training, use the model for inference:
```python
from ultralytics import YOLO

model = YOLO('runs/detect/card_game_yolo/weights/best.pt')
results = model.predict('path/to/image.png', conf=0.5)

# Access results
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    print(boxes.xyxy)     # box coordinates in (x1, y1, x2, y2) format
    print(boxes.conf)     # confidence scores
    print(boxes.cls)      # class ids
```

## Categories

The model detects the following classes:
- **Symbols**: anchor, arrow_down, arrow_left, arrow_right, coin, kraken, map, parrot, rat, rum, shark, spyglass
- **Quarters**: quarter (card quadrants)

## Dataset Format

The training script expects JSON annotations in the following format:
```json
{
  "image": {
    "id": 0,
    "width": 1472,
    "height": 1540,
    "file_name": "sample_00000.png"
  },
  "annotations": [
    {
      "id": 0,
      "category_id": 8,
      "category_name": "quarter",
      "bbox": [x, y, width, height]
    }
  ],
  "categories": [
    {
      "id": 0,
      "name": "anchor",
      "supercategory": "symbol"
    }
  ]
}
```

## Output Structure

After running the training script, the following structure will be created:
```
yolo_dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml

runs/detect/card_game_yolo/
├── weights/
│   ├── best.pt
│   └── last.pt
├── args.yaml
├── results.csv
└── *.png (training plots)
```

## Performance Monitoring

Training metrics are saved in `runs/detect/card_game_yolo/results.csv` and include:
- Box loss
- Class loss
- DFL loss
- Precision
- Recall
- mAP50
- mAP50-95

Plots are automatically generated for:
- Training/validation losses
- Precision-recall curves
- Confusion matrix
- F1 scores

## Tips for Better Results

1. **More data**: Generate more training samples for better accuracy
2. **Augmentation**: YOLO automatically applies augmentations, but you can customize in `train()` method
3. **Model size**: Use larger models ('s', 'm', 'l', 'x') if you have more compute resources
4. **Batch size**: Increase if you have more GPU memory
5. **Image size**: Use 1024 or higher for detecting small objects
6. **Epochs**: Increase if validation metrics are still improving

## Troubleshooting

- **Out of memory**: Reduce `batch_size` or use smaller model
- **Slow training**: Use GPU by ensuring CUDA is installed
- **Poor accuracy**: Check if annotations are correct, increase dataset size, or train longer

