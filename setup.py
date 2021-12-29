
from setuptools import setup, find_packages

setup(
    name="weirdneuralnet",
    version="0.0.2",
    author="Dane Howard",
    author_email="dane.a.howard@gmail.com",
    description="A small neural net experimentation framework",
    url="https://github.com/mirrord/weirdneuralnet",
    install_requires=[
        'matplotlib',
        'scipy',
        'scikit-learn',
        'tqdm',
        'requests',
        'pyqt5',
        'cupy' #TODO: create a way to install appropriate cupy from pre-built binaries
                #this will build CuPy from source which takes FOREVER
    ],
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