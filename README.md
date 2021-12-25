<div id="top"></div>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <!-- <a href="https://github.com/manassharma07/crysx_nn">
    <img src="logo_crysx_nn.png" alt="Logo" width="320" height="200">
  </a> -->

<h3 align="center">crysx_nn</h3>

  <p align="center">
    A simplistic and efficient pure-python neural network library from Phys Whiz.
    <br />
    <a href="https://github.com/manassharma07/crysx_nn"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/github_username/repo_name">View Demo</a>
    ·
    <a href="https://github.com/manassharma07/crysx_nn/issues">Report Bug</a>
    ·
    <a href="https://github.com/manassharma07/crysx_nn/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com)

Neural networks are an integral part of machine learning.
The project provides an easy-to-use, yet efficient implementation that can be used in your projects or for teaching/learning purposes.

The library is written in pure-python with some optimizations using numpy, opt_einsum and numba. 

The goal was to create a framework that is efficient yet easy to understand, so that everyone can see and learn about what goes inside a neural network. After all, the project did spawn from a semester project on [CP_IV: Machine Learning course](https://friedolin.uni-jena.de/qisserver/rds?state=verpublish&status=init&vmfile=no&moduleCall=webInfo&publishConfFile=webInfo&publishSubDir=veranstaltung&veranstaltung.veranstid=187951) at the [University of Jena, Germany](https://www.uni-jena.de/en/).


<p align="right">(<a href="#top">back to top</a>)</p>



### Built With

* [NumPy](https://numpy.org)
* [numba](https://numba.pydata.org)
* [opt_einsum](https://optimized-einsum.readthedocs.io/en/stable/)
* [matplotlib](https://matplotlib.org)
* [nnv](https://pypi.org/project/nnv/)


<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple example steps.

### Prerequisites

You need to have `python3` installed along with `pip`.

### Installation

There are many ways to install `crysx_nn`

1. Install the release (stable) version from [PyPi](https://example.com)
    ```sh
    pip install crysx_nn
    ```
2. Install the latest development version, by cloning the git repo and installing it. 
   This requires you to have `git` installed.
   ```sh
   git clone https://github.com/manassharma07/crysx_nn.git
   cd crysx_nn
   pip install .
   ```
3. Install the latest development version without `git`.
   ```sh
   pip install --upgrade https://github.com/manassharma07/crysx_nn/tarball/main
   ```

Check if the installation was successful by running python shell and trying to import the package 
```sh
python3
```
```python
Python 3.7.11 (default, Jul 27 2021, 07:03:16) 
[Clang 10.0.0 ] :: Anaconda, Inc. on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import crysx_nn
>>> crysx_nn.__version__
'0.1.0'
>>> 
```

   Finally download the example script ([here](https://github.com/manassharma07/crysx_nn/blob/main/examples/Simulating_Logic_Gates.py)) for simulating logic gates like AND, XOR, NAND, and OR,
   and try running it
   ```sh
   python Simluating_logic_gates.py
   ```

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

The most important thing for using this library properly is to use 2D NumPy arrays for defining the inputs and exoected outputs (targets) for a network. 1D arrays for inputs and targets are not supported and will result in an error.

For example, let us try to simulate the logic gate AND. The AND gate takes two input bits and returns a single input bit.
The bits can take a value of either 0 or 1. The AND gate returns 1 only if both the inputs are 1, otherwise it returns 0.

The truth table of the AND gate is as follows

x1 | x2 | output 
--- | --- | --- 
0 | 0 | 0 
0 | 1 | 0 
1 | 0 | 0 
1 | 1 | 1 

The four possible set of inputs are 
```python
   inputs = np.arrary([])
```

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [ ] Weights and biases initialization
- [ ] More activation functions
    - [ ] Identity, LeakyReLU, Tanh, etc.
- [ ] More loss functions
    - [ ] categorical cross entropy, and others
- [ ] Optimization algorithms apart from Stochastic Gradient Descent, like ADAM, RMSprop, etc.
- [ ] Implement regularizers
- [ ] Batch normalization
- [ ] Dropout
- [ ] Early stopping
- [ ] A `predict` function that returns the output of the last layer and the loss/accuracy
- [ ] Some metric functions, although there is no harm in using `sklearn` for that 

See the [open issues](https://github.com/manassharma07/crysx_nn/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Manas Sharma - [@manassharma07](https://twitter.com/manassharma07) - feedback@bragitoff.com

Project Link: [https://github.com/manassharma07/crysx_nn](https://github.com/manassharma07/crysx_nn)

Project Documentation: [https://bragitoff.com](https://bragitoff.com/)

Blog: [https://bragitoff.com](https://bragitoff.com/)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* Lecture notes by [Prof. Dr. Bernd Bürgmann](https://www.physik.uni-jena.de/en/bruegmann)
* []()
* []()

<p align="right">(<a href="#top">back to top</a>)</p>

## Citation
If you use this library and would like to cite it, you can use:
```
 M. Sharma, "CrysX-NN: Neural Network libray", 2021. [Online]. Available: https://github.com/manassharma07/crysx_nn. [Accessed: DD- Month- 20YY].
```
or:
```
@Misc{,
  author = {Manas Sharma},
  title  = {CrysX-NN: Neural Network libray},
  month  = december,
  year   = {2021},
  note   = {Online; accessed <today>},
  url    = {https://github.com/manassharma07/crysx_nn},
}
```
<p align="right">(<a href="#top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/manassharma07/crysx_nn.svg?style=for-the-badge
[contributors-url]: https://github.com/manassharma07/crysx_nn/contributors
[forks-shield]: https://img.shields.io/github/forks/manassharma07/crysx_nn.svg?style=for-the-badge
[forks-url]: https://github.com/manassharma07/crysx_nn/network/members
[stars-shield]: https://img.shields.io/github/stars/manassharma07/crysx_nn.svg?style=for-the-badge
[stars-url]: https://github.com/manassharma07/crysx_nn/stargazers
[issues-shield]: https://img.shields.io/github/issues/manassharma07/crysx_nn.svg?style=for-the-badge
[issues-url]: https://github.com/manassharma07/cysx_nn/issues
[license-shield]: https://img.shields.io/github/license/manassharma07/crysx_nn.svg?style=for-the-badge
[license-url]: https://github.com/manassharma07/cysx_nn/blob/main/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/manassharma07
[product-screenshot]: logo_crysx_nn.png