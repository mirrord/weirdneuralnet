
from setuptools import setup, find_packages

__version__ = "0.0.5"

with open("requirements.txt", 'r') as f:
    requirements = f.readlines()    

# 'cupy' #TODO: create a way to install appropriate cupy from pre-built binaries
# just adding 'cupy' to reqs will build CuPy from source which takes FOREVER
# TODO: default back to numpy if cupy can't be installed

setup(
    name="weirdneuralnet",
    version=__version__,
    author="Dane Howard",
    author_email="dane.a.howard@gmail.com",
    description="A small neural net experimentation framework",
    url="https://github.com/mirrord/weirdneuralnet",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.6",
    test_suite="unit_tests"
)