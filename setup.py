from setuptools import find_packages, setup

setup(
    name="privacy-eval",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "pandas",
    ],
    dependency_links=[],
)
