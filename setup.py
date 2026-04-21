from setuptools import setup, find_packages
from pathlib import Path

requirements = Path("requirements.txt").read_text().strip().splitlines()

setup(
    name="causallens",
    version="0.1.0",
    author="Stefano Casafranca",
    author_email="stefano.casafranca@ucalgary.ca",
    description="Open-source toolkit for causal autonomy auditing of recommender systems",
    long_description=Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/stefanocasafranca/CausalLens",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
