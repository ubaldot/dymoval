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

Whether your model is a Deep Neural Network, an ODE, or something more
complex, and regardless of whether you use Modelica or Simulink as your
modeling tool, Dymoval has you covered. Simply provide Dymoval with
measurements and model-generated data, and it will deliver a comprehensive
model quality evaluation, including r-squared fit, residuals norms, and
coverage region.

<div align="center"
	<br>
	<br>
<img src="https://github.com/VolvoGroup/dymoval/blob/main/docs/source/figures/DymovalNutshell.svg" data-canonical-src="[https://github.com/VolvoGroup/dymoval/blob/main/docs/source/figures/DymovalNutshell.svg](https://github.com/VolvoGroup/dymoval/blob/main/docs/source/DymovalNutshell.svg)" width="600" class="center"  />
	<br>
	<br>
	<br>
</div>

If you are tracking your models changes in a CI/CD environment, then _Dymoval_
API can be easily used to run tests in Jenkins or GitHub Actions pipelines.

Dymoval also provides essential functions for handling measurement data,
addressing common issues such as noise, missing data, and varying sampling
intervals.

## Why dymoval?

Simulation results frequently deviate significantly from real-world
measurements, leading to a growing skepticism towards simulation models.
_Dymoval_ is dedicated to bridging this gap, aiming to restore confidence in
simulation accuracy and reliability.

Dymoval specializes in model validation, offering robust solutions for a
variety of models, including MIMO (Multiple Input Multiple Output) and stiff
models, all in an easy and comprehensible manner. Additionally, Dymoval
provides a comprehensive toolbox designed to handle real-world measurement
data, which often comes with challenges such as noise, missing data, and
varying sampling intervals. This ensures that your models are not only
validated but also capable of accurately reflecting real-world conditions.

## Main Features

**Model validation**

* Validation metrics:
  * R-square fit
  * Residuals auto-correlation statistics
  * Input-Residuals cross-correlation statistics
* Coverage region
* MIMO models
* Independence of the modeling tool used.
* API suitable for model unit-tests

**Measurement data analysis and manipulation**

* Time and frequency analysis
* Easy plotting
* Missing data handling
* Linear filtering
* Means and offsets removal
* Re-sampling
* Physical units

## Installation

Dymoval exists on both `pip` and `conda`, so you can choose between the
following:

    pip install dymoval
    conda install -c conda-forge dymoval

## Getting started

Suppose you want to validate a model with $p$ inputs and $q$ outputs and have
the corresponding measurement data available.

Do as it follows:

* Feed your model with the measurement data corresponding to the model input,
* Collect the simulated out `y_sim` and arrange them along with the measured
  input `u_meas`, measured out `y_meas` and $Nxq$, $Nxp$ and $Nxq$
  `np.ndarray`, respectively.
* Call
  `validate_models(measured_in=u_meas, measured_out=y_meas, simulated_out=y_sim, sampling\_period = sampling\_period`,
  where `sampling\_period` is the signals sampling period, and see the
  results.

You should see something like the following:

```
Input whiteness (abs_mean-max)      0.3532
R-Squared (%)                      65.9009
Residuals whiteness (abs_mean-max)  0.1087
Input-Res whiteness (abs_mean-max)  0.2053

         My_Model
Outcome: PASS
```

The rule-of-thumb is that the R-squared index shall be as high as possible,
while the various whiteness metrics should be as close to zero as possible.

If your results are not satisfactory, don't worry. It might not be the model
that's at fault, but rather the validation procedure may need more tweaking.
No need to panic... yet! :)

Take a look at the tutorial that you can open with the following:

```
import dymoval as dmv
dmv.open_tutorial()
```

The above commands create a `dymoval_tutorial` folder containing all the files
needed to run the tutorial in your `home` folder. All you have to do is to run
Jupyter notebook named `dymoval_tutorial.ipynb`. You need an app for opening
`.ipynb` files.

Finally, if you want a deeper understanding on _Dymoval_ and on model validation
in general, check out the [docs](https://ubaldot.github.io/dymoval/).

## License

_Dymoval_ is licensed under
[BSD 3](https://github.com/ubaldot/dymoval/blob/main/LICENSE) license.
