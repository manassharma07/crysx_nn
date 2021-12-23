import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "crysx_nn",
    version = "0.0.1",
    author = "Manas Sharma",
    author_email = "feedback@bragitoff.com",
    description = ("A simplistic and efficient pure-python neural network library from Phys Whiz."),
    license = "MIT",
    keywords = ["neural network", "pure python", "crysx", "numba nn", "machine learning", "ML", "deep learning", "deepL", "MLP", "perceptron","phys whiz","manas sharma","bragitoff","crysx"],
    url = "http://bragitoff.com",
    packages=['crysx_nn'],
    long_description=read('README.md'),
    install_requires=['numba>=0.54.1',
                      'numpy==1.19.2', 
                      'autograd',
                      'tqdm',
                      'opt_einsum',
                      'nnv',
                      'matplotlib',
                      'numexpr'                    
                      ],
    classifiers=[
        "Development Status :: 1 - Alpha",
        "Topic :: Deep Learning - Neural networks",
        "Intended Audience :: Science/Research",
        "License ::  MIT License",
        "Programming Language :: Python :: >3.5",
        "Operating System :: Windows, MacOS, Linux", 
    ],
)