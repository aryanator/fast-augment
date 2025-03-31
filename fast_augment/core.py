import os
import cv2
import numpy as np
import random
from torchvision.datasets import CIFAR10
from tqdm import tqdm
from typing import Optional, Union, List, Tuple

class FastAugment:
    def __init__(self, preset: str = "simple"):
        self.preset = preset
        self.augmentations = self._get_augmentations()

    def _get_augmentations(self) -> List[callable]:
        """Returns list of augmentation functions based on preset"""
        if self.preset == "simple":
            return [
                self._random_horizontal_flip,
                self._random_rotate
            ]
        elif self.preset == "advanced":
            return [
                self._random_horizontal_flip,
                self._random_rotate,
                self._random_cutout,
                self._random_brightness_contrast
            ]
        else:
            raise ValueError(f"Unknown preset: {self.preset}")

    def _random_horizontal_flip(self, image: np.ndarray) -> np.ndarray:
        """Random horizontal flip with 50% probability"""
        if random.random() < 0.5:
            return cv2.flip(image, 1)
        return image

    def _random_rotate(self, image: np.ndarray) -> np.ndarray:
        """Random rotation between -30 and 30 degrees"""
        if random.random() < 0.5:
            angle = random.uniform(-30, 30)
            h, w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            return cv2.warpAffine(image, M, (w, h))
        return image

    def _random_cutout(self, image: np.ndarray) -> np.ndarray:
        """Random cutout with 8 holes of max size 8x8 (50% probability)"""
        if random.random() < 0.5:
            h, w = image.shape[:2]
            for _ in range(8):
                y = random.randint(0, h)
                x = random.randint(0, w)
                size = random.randint(1, 8)
                y1 = max(0, y - size//2)
                y2 = min(h, y + size//2)
                x1 = max(0, x - size//2)
                x2 = min(w, x + size//2)
                image[y1:y2, x1:x2] = 0
        return image

    def _random_brightness_contrast(self, image: np.ndarray) -> np.ndarray:
        """Random brightness/contrast adjustment (30% probability)"""
        if random.random() < 0.3:
            # Brightness
            beta = random.uniform(-0.2, 0.2) * 255
            # Contrast
            alpha = random.uniform(0.8, 1.2)
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return image

    def augment_image(self, image: np.ndarray) -> np.ndarray:
        """Apply all augmentations to single image"""
        for aug in self.augmentations:
            image = aug(image)
        return image

    def augment_dataset(
        self,
        dataset: Union[str, list],
        output_dir: Optional[str] = None,
        target_size: Optional[int] = None
    ) -> List[Tuple[np.ndarray, int]]:
        """
        Augment entire dataset
        Args:
            dataset: Either path to images or list of (image, label) tuples
            output_dir: If provided, saves augmented images here
            target_size: Desired number of output samples
        Returns:
            List of (augmented_image, label) tuples
        """
        # Load data if path provided
        if isinstance(dataset, str):
            from torchvision.datasets import ImageFolder
            dataset = ImageFolder(dataset)
            images = [np.array(x[0]) for x in dataset]
            labels = [x[1] for x in dataset]
        else:
            images = [x[0] for x in dataset]
            labels = [x[1] for x in dataset]

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        augmented = []
        original_size = len(images)
        target_size = target_size or original_size
        
        with tqdm(total=target_size) as pbar:
            while len(augmented) < target_size:
                for i in range(original_size):
                    if len(augmented) >= target_size:
                        break
                        
                    img = images[i].copy()
                    label = labels[i]
                    
                    # Apply augmentations
                    aug_img = self.augment_image(img)
                    
                    if output_dir:
                        save_path = os.path.join(output_dir, f"aug_{i}_{len(augmented)}.png")
                        cv2.imwrite(save_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                    
                    augmented.append((aug_img, label))
                    pbar.update(1)
        
        return augmented