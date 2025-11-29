"""Setup script for GMV Forecasting MLOps package"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="gmv-forecasting-mlops",
    version="1.0.0",
    author="GMV Forecasting Team",
    author_email="your.email@example.com",
    description="Production-ready GMV forecasting with hybrid SARIMAX + Prophet model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/gmv-forecasting-mlops",
    packages=find_packages(exclude=["tests", "notebooks", "deployment"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "black>=23.11.0",
            "flake8>=6.1.0",
            "mypy>=1.7.1",
            "pre-commit>=3.5.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "gmv-prepare-data=src.data.prepare_data:main",
            "gmv-train=src.models.train:main",
            "gmv-evaluate=src.evaluation.evaluate:main",
            "gmv-api=src.api.main:main"
        ]
    },
    include_package_data=True,
    zip_safe=False
)

