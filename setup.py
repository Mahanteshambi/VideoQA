from setuptools import setup, find_packages

setup(
    name="videoqa",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # Add your dependencies here if needed
    ],
) 