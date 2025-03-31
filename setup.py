from setuptools import setup, find_packages

setup(
    name="fast-augment",
    version="0.1.2",
    packages=find_packages(),
    install_requires=["albumentations>=1.0.0", "torchvision", "opencv-python", "tqdm"],
)