from setuptools import find_packages, setup

setup(
    name="jupiter_gym",
    version="0.0.1",
    install_requires=["gym", "numpy", "solana"],
    packages=find_packages(),
)
