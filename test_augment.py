from fast_aug import FastAugment
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Test with built-in sample image
def create_sample_image():
    """Generate a simple color gradient test image"""
    img = np.zeros((128,128,3), dtype=np.uint8)
    for i in range(128):
        img[:,i] = [i*2, 255-i*2, i]
    return img

# Run test
original = create_sample_image()
augmenter = FastAugment(preset="advanced")
augmented = augmenter.augment_image(original.copy())

# Display results
plt.figure(figsize=(10,5))
plt.subplot(121)
plt.title("Original")
plt.imshow(original)
plt.subplot(122)
plt.title("Augmented")
plt.imshow(augmented)
plt.show()