# FastAugment 🚀

One-command image augmentation for computer vision pipelines. Apply transformations with a single function call.

[![PyPI version](https://img.shields.io/pypi/v/fastaugment)](https://pypi.org/project/fastaugment/)
[![Python versions](https://img.shields.io/pypi/pyversions/fastaugment)](https://pypi.org/project/fastaugment/)
[![License](https://img.shields.io/pypi/l/fastaugment)](LICENSE)
[![PyPI Downloads](https://static.pepy.tech/badge/fast-augment)](https://pepy.tech/projects/fast-augment)

## Features

- 🛠️ **Preset-based augmentations** - Choose between "simple" or "advanced" augmentation strategies
- 🖼️ **Supports multiple input types** - Works with image paths, numpy arrays, and PyTorch datasets
- ⚡ **Efficient processing** - Optimized OpenCV backend
- 📁 **Automatic saving** - Optionally save augmented images to directory

## Installation

```bash
pip install fastaugment
```

## Quick Start

### Basic Usage

```python
from fastaugment import FastAugment
import cv2

# Load an image
image = cv2.imread("image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Initialize augmenter
augmenter = FastAugment(preset="advanced")

# Augment single image
augmented_image = augmenter.augment_image(image)
```

### Dataset Augmentation

```python
from torchvision.datasets import CIFAR10

# Load dataset
dataset = CIFAR10(root="./data", train=True)

# Augment entire dataset
augmenter = FastAugment(preset="advanced")
augmented_data = augmenter.augment_dataset(
    dataset=dataset,
    output_dir="./augmented_data",
    target_size=10000
)
```

## Presets

| Preset     | Transformations                          |
|------------|-----------------------------------------|
| `simple`   | Horizontal flips, rotations             |
| `advanced` | Adds cutout and brightness/contrast     |

## Advanced Configuration

Customize individual augmentation probabilities:

```python
# Coming in v1.1 (create feature request!)
```

## Documentation

Full API reference available at [fastaugment.readthedocs.io](https://fastaugment.readthedocs.io)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Aryan Patil - aryanator01@gmail.com
```

