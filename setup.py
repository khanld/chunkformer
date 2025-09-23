#!/usr/bin/env python
"""
Setup script for ChunkFormer: Masked Chunking Conformer For Long-Form Speech Transcription
"""

import os
from setuptools import setup, find_packages

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    with open(readme_path, "r", encoding="utf-8") as f:
        return f.read()

# Read requirements from requirements.txt
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    with open(requirements_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="chunkformer",
    version="0.1.0",
    author="khanhld",
    author_email="your-email@example.com",
    description="ChunkFormer: Masked Chunking Conformer For Long-Form Speech Transcription",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/chunkformer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "torchaudio>=0.9.0",
        "transformers>=4.20.0",
        "PyYAML>=5.4.0",
        "pandas>=1.3.0",
        "tqdm>=4.62.0",
        "jiwer>=2.3.0",
        "colorama>=0.4.4",
        "pydub>=0.25.0",
        "Pillow>=8.3.0",
        "sentencepiece>=0.1.96",
        "textgrid>=1.5.0",
        "huggingface_hub>=0.10.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "isort>=5.9.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "chunkformer-decode=chunkformer.decode:main",
        ],
    },
    include_package_data=True,
    package_data={
        "chunkformer": [
            "chunkformer-large-vie/*",
            "chunkformer-large-vie-hf/*",
            "data/*",
            "docs/*",
        ],
    },
    keywords=[
        "speech-recognition",
        "asr",
        "transformer",
        "conformer",
        "pytorch",
        "long-form-audio",
        "machine-learning",
        "deep-learning",
    ],
    project_urls={
        "Bug Reports": "https://github.com/your-username/chunkformer/issues",
        "Source": "https://github.com/your-username/chunkformer",
        "Documentation": "https://github.com/your-username/chunkformer/blob/main/README.md",
        "Paper": "https://github.com/your-username/chunkformer/blob/main/docs/paper.pdf",
    },
)