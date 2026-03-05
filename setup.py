from setuptools import setup, find_packages

setup(
    name="banglanlpkit",
    version="0.1",
    packages=find_packages(),  # will find banglanlpkit and subpackages
    install_requires=[
        # list dependencies if any, e.g., "requests", "beautifulsoup4"
    ],
    python_requires=">=3.8",
)