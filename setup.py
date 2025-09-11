from setuptools import setup, find_packages

setup(
    name="inclusivePackage",
    version="0.1.0",
    packages=find_packages(where="inclusivePackage"),
    package_dir={"": "inclusivePackage"},
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.2.3",
        "scikit-learn>=1.6.1",
        "matplotlib>=3.10.1",
        "python-dateutil>=2.9.0",
        "pytz>=2025.2",
        "setuptools>=75.8.2",
    ],
    python_requires=">=3.10",
)
