from setuptools import setup, find_packages

setup(
    name="RefGuidedINR",
    version="0.1.0",
    description="Reference-Guided Arbitrary-Scale Super-Resolution via Implicit Neural Representation",
    author="",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.10.0",
        "torchvision>=0.11.0",
        "numpy>=1.21.0",
        "Pillow>=8.3.0",
        "PyYAML>=5.4.0",
        "tqdm>=4.62.0",
        "lpips>=0.1.4",
        "scikit-image>=0.18.0",
        "einops>=0.4.0",
        "tensorboard>=2.7.0",
    ],
)
