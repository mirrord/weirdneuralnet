
from setuptools import setup, find_packages

setup(
    name="weirdneuralnet",
    version="0.0.1",
    install_requires=[
        'matplotib',
        'scipy',
        'requests',
        'cupy'
    ],
    packages=find_packages(exclude="test_scripts"),
    
)