# PyTorch Image Classifier

A deep learning image classification tool built with PyTorch that allows you to train models on custom image datasets and make predictions.

## Overview

This project implements a transfer learning approach to image classification by fine-tuning pre-trained convolutional neural networks (CNNs) on custom datasets. It includes utilities for data processing, model training, checkpointing, and prediction.

## Features

- Train image classifiers using pre-trained architectures (VGG16, VGG13, ResNet50)
- Custom data preprocessing pipeline with augmentation
- Model checkpointing and loading
- Command-line interface for training and prediction
- Support for GPU acceleration
- Early stopping functionality to prevent overfitting
- Validation during training to monitor performance

## Project Structure

```
├── data_utils.py     # Data loading and preprocessing utilities
├── model_utils.py    # Model setup, training, and evaluation functions
├── train.py          # Command-line script for training models
├── predict.py        # Command-line script for making predictions
└── checkpoints/      # Directory for saved model checkpoints
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pytorch-image-classifier.git
cd pytorch-image-classifier
```

2. Install dependencies:
```bash
pip install torch torchvision numpy pillow matplotlib
```

## Usage

### Training a Model

To train a new model on your dataset:

```bash
python train.py data_directory --save_dir checkpoints --arch vgg16 --learning_rate 0.001 --hidden_units 4096 --epochs 3 --gpu
```

Arguments:
- `data_dir`: Path to the data directory. Should have train, valid, and test subdirectories
- `--save_dir`: Directory to save checkpoints (default: 'checkpoints')
- `--arch`: Model architecture (choices: vgg16, vgg13, resnet50, default: vgg16)
- `--learning_rate`: Learning rate (default: 0.001)
- `--hidden_units`: Number of hidden units (default: 4096)
- `--epochs`: Number of training epochs (default: 3)
- `--gpu`: Use GPU for training if available

### Making Predictions

To predict the class of an image using a trained model:

```bash
python predict.py path/to/image checkpoint_file --top_k 5 --category_names cat_to_name.json --gpu
```

Arguments:
- `image_path`: Path to the input image
- `checkpoint`: Path to a saved model checkpoint
- `--top_k`: Return top K most likely classes (default: 1)
- `--category_names`: JSON file mapping categories to real names
- `--gpu`: Use GPU for inference if available

## Data Directory Structure

Your data directory should be organized as follows:

```
data_directory/
  ├── train/
  │   ├── class_1/
  │   │   ├── image1.jpg
  │   │   └── ...
  │   ├── class_2/
  │   │   └── ...
  ├── valid/
  │   ├── class_1/
  │   │   └── ...
  │   ├── class_2/
  │   │   └── ...
  └── test/
      ├── class_1/
      │   └── ...
      ├── class_2/
      │   └── ...
```

## Advanced Features

### Early Stopping

The `EarlyStopping` class in `model_utils.py` can be used to prevent overfitting:

```python
from model_utils import EarlyStopping, train

early_stopping = EarlyStopping(patience=5, min_delta=0.01)
train(model, trainloader, validloader, criterion, optimizer, early_stopping=early_stopping)
```

### Image Processing

The `process_image` function in `data_utils.py` handles all preprocessing required for model input:

```python
from data_utils import process_image
processed_image = process_image('path/to/image.jpg')
```

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
