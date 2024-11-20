<div align="center">

<img src="https://github.com/VolvoGroup/dymoval/blob/main/docs/source/figures/DymovalLogo.svg" data-canonical-src="[https://github.com/VolvoGroup/dymoval/blob/main/docs/source/figures/DymovalLogo.svg](https://github.com/VolvoGroup/dymoval/blob/main/docs/source/figures/DymovalLogo.svg)" width="800" class="center" />

</div>

### Build status

![pipeline](https://github.com/VolvoGroup/dymoval/actions/workflows/pipeline.yml/badge.svg)
![coverage badge](./coverage.svg)

### Tools

[![Build - pdm](https://img.shields.io/badge/build-pdm-blueviolet)](https://pdm.fming.dev/latest/)
[![code check - flake8](https://img.shields.io/badge/checks-flake8-green.svg)](https://pypi.org/project/flake8)
[![types - Mypy](https://img.shields.io/badge/types-mypy-orange.svg)](https://github.com/python/mypy)
[![test - pytest](https://img.shields.io/badge/tests-pytest-brightgreen.svg)](https://github.com/pytest-dev/pytest)
[![code style - black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![docs - sphinx](https://img.shields.io/badge/docs-sphinx-blue.svg)](https://github.com/sphinx-doc/sphinx)

---

## What is it?

**Dymoval** (**Dy**namic **Mo**del **Val**idation) is a Python package for
analyzing _measurements data_ and validate _models_.

It does not matter if your model is a Deep Neural Network, an ODE or something
more complex, nor is important if you use Modelica or Simulink or whatever as
modeling tool. All you need to do is to feed _Dymoval_ with real-world
measurements and model-generated data and you will get a model quality
evaluation in r-squared fit, residuals norms and coverage region.

<div align="center"
	<br>
	<br>
<img src="https://github.com/VolvoGroup/dymoval/blob/main/docs/source/figures/DymovalNutshell.svg" data-canonical-src="[https://github.com/VolvoGroup/dymoval/blob/main/docs/source/figures/DymovalNutshell.svg](https://github.com/VolvoGroup/dymoval/blob/main/docs/source/DymovalNutshell.svg)" width="600" class="center"  />
	<br>
	<br>
	<br>
</div>

If you are developing your models in a CI/CD environment, then _Dymoval_ can
help you in deciding if merging or not the latest model changes. In-fact,
_Dymoval_ functions can also be included in development pipelines scripts, so
the whole CI/CD process can be fully automatized.

_Dymoval_ finally provides you with some essential functions for dealing with
dataset analysis and manipulation.

Although the tool has been thought with engineers in mind, no one prevents you
to use it in other application domains.

## Why dymoval?

There plenty of amazing packages out there like _matplotlib_, _pandas_,
_numpy_, _scipy_, etc for analyzing data, compute statistics, and so on, but
they are huge and the plethora of functionalities they offer may be
overwhelming.

_Dymoval_ has been built on top of these tools and it aims at providing an
extremely easy and intuitive API that shall serve most of the tasks an
engineer typically face in his/her daily job in a simple and comprehensive
way.

However, _Dymoval_ leaves the door open: most of the functions return objects
that can be used straight away with the aforementioned tools without any extra
steps. Hence, if you need more power, you always get an object that can be
immediately handled by some other more powerful tool while using _Dymoval_.

## Main Features

**Measurement data analysis and manipulation**

* Time and frequency analysis
* Easy plotting
* Missing data handling
* Linear filtering
* Means and offsets removal
* Re-sampling
* Physical units

**Model validation**

* Validation metrics:
  * R-square fit
  * Residuals auto-correlation statistics
  * Input-Residuals cross-correlation statistics
* Coverage region
* Enable model unit-tests
* Work for both SISO and MIMO models
* Modeling tool independence
* Easily integrate with CI/CD pipelines.

The residuals x-correlation statistics are expressed in terms of weighted
quadratic forms, thus allowing model quality evaluation in terms of known
tests such as Ljung-Box, Box-Pierce, etc.

## Installation

Dymoval exists on both `pip` and `conda`, so you can choose between the
following:

    pip install dymoval
    conda install -c conda-forge dymoval

## Getting started

If you are already familiar with model validation, then the best way to get
started with dymoval is to run the tutorial scripts that you can open with

    import dymoval as dmv
    dmv.open_tutorial()

This create a `dymoval_tutorial` folder containing all the files needed to run
the tutorial in your `home` folder. All you have to do is to run Jupyter
notebook named `dymoval_tutorial.ipynb`. You need an app for opening `.ipynb`
files.

For more info on what is model validation and what is implemented in _dymoval_
along with the _dymoval_ complete API, you can check the
[docs](https://ubaldot.github.io/dymoval/).

## License

Dymoval is licensed under
[BSD 3](https://github.com/ubaldot/dymoval/blob/main/LICENSE) license.
