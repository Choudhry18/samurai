import os
from setuptools import find_packages, setup

# Package metadata
NAME = "samurai"
VERSION = "0.1.0"
DESCRIPTION = "SAMURAI: SAM for zero-shot visual tracking with Motion-Aware Memory"
URL = "https://github.com/yangchris11/samurai"
AUTHOR = "Cheng-Yen Yang"
AUTHOR_EMAIL = "cycyang@uw.edu"
LICENSE = "Apache 2.0"

# Read the contents of README file
with open("README.md", "r", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

# Required dependencies - include all SAM2 dependencies plus SAMURAI's specific ones
REQUIRED_PACKAGES = [
    "torch>=2.3.1",
    "torchvision>=0.18.1",
    "numpy>=1.24.4",
    "tqdm>=4.66.1",
    "hydra-core>=1.3.2",
    "iopath>=0.1.10",
    "pillow>=9.4.0",
    "opencv-python",
    "scipy",
    "loguru",
    "matplotlib>=3.7",
    "pandas",
    "huggingface_hub",
]

# Setup configuration
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url=URL,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license=LICENSE,
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'samurai': ['checkpoints/download_ckpts.sh'],
    },
    install_requires=REQUIRED_PACKAGES,
    python_requires=">=3.10.0",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
