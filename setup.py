from setuptools import setup, find_packages

setup(
    name="triton-test",
    version="0.1.0",
    description="Testing project for Triton language",
    packages=find_packages(),
    install_requires=[
        "torch",
        "pytest",
        "numpy",
    ],
    python_requires=">=3.8",
)