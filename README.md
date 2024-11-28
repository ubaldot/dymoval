<div align="center">

<img src="https://github.com/VolvoGroup/dymoval/blob/main/docs/source/figures/DymovalLogo.svg" data-canonical-src="[https://github.com/VolvoGroup/dymoval/blob/main/docs/source/figures/DymovalLogo.svg](https://github.com/VolvoGroup/dymoval/blob/main/docs/source/figures/DymovalLogo.svg)" width="800" class="center" />

</div>

### Build status

![pipeline](https://github.com/VolvoGroup/dymoval/actions/workflows/pipeline.yml/badge.svg)
![coverage badge](./coverage.svg)

### Tools

[![types - mypy](https://img.shields.io/badge/types-mypy-orange.svg)](https://github.com/python/mypy)
[![test - pytest](https://img.shields.io/badge/tests-pytest-brightgreen.svg)](https://github.com/pytest-dev/pytest)
[![code style - ruff](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/astral-sh/ruff)
[![docs - sphinx](https://img.shields.io/badge/docs-sphinx-blue.svg)](https://github.com/sphinx-doc/sphinx)

---

## What is it?

_Dymoval_ is a Python package for _analyzing measurement data_ and _validating
models_.

_Dymoval_ validates models based only on _measurements_ and _simulated data_,
and it is completely independent of the used modeling tool. That means that it
does not matter if a model has been developed with Simulink, Modelica, etc.,
_dymoval_ will only look at the produced data from the model.

<div align="center"
	<br>
	<br>
<img src="https://github.com/VolvoGroup/dymoval/blob/main/docs/source/figures/DymovalNutshell.svg" data-canonical-src="[https://github.com/VolvoGroup/dymoval/blob/main/docs/source/figures/DymovalNutshell.svg](https://github.com/VolvoGroup/dymoval/blob/main/docs/source/DymovalNutshell.svg)" width="600" class="center"  />
	<br>
	<br>
	<br>
</div>

If you are tracking your models changes in a CI/CD environment, then _dymoval_
API can be easily used to run tests in Jenkins or GitHub Actions pipelines as
it enables unit-testing on models.

Finally, _dymoval_ provides a number of functions for for handling
measurements data, addressing common issues such as noise, missing data, and
varying sampling intervals.

## Installation

Dymoval exists on both `pip` and `conda`, so you can choose between the
following:

```
pip install dymoval
conda install -c conda-forge dymoval
```

## Getting started

Suppose you want to validate a model and you have the simulated out `y_sim`
the measured input `u_meas`, and the measured out `y_meas` time-series sampled
with period `sampled_period`. Just call the following function:

```
from dymoval.validation import validate_models

validate_models(
    measured_in=u_meas,
    measured_out=y_meas,
    simulated_out=y_sim,
    sampling_period = sampling_period
)
```

to get something like the following:

```
  Input whiteness (abs_mean-max)      0.3532
  R-Squared (%)                      65.9009
  Residuals whiteness (abs_mean-max)  0.1087
  Input-Res whiteness (abs_mean-max)  0.2053

           My_Model
  Outcome: PASS
```

Congrats! Your model passed the test!

But what if the test didn't pass? Don't worry, it might not be the model's
fault.

For example, you could be dealing with noisy measurements, over-sampled
signals, missing data, and other factors that might affect the results. Take a
look at the tutorial to learn how to address such issues:

```
  import dymoval as dmv
  dmv.open_tutorial()
```

The above commands create a `dymoval_tutorial` folder containing all the files
needed to run the tutorial in your `home` folder. All you have to do is to run
Jupyter notebook named `dymoval_tutorial.ipynb`. You need an app for opening
`.ipynb` files.

And if you want to discover more on model validation and _Dymoval_ check out
the [docs](https://ubaldot.github.io/dymoval/).

## Main Features

**Model validation**

- Validation metrics:
  - R-square fit
  - Residuals auto-correlation statistics
  - Input-Residuals cross-correlation statistics
- Coverage region
- MIMO models
- Independence of the modeling tool used.
- API suitable for model unit-tests

**Measurement data analysis and manipulation**

- Time and frequency analysis
- Easy plotting
- Missing data handling
- Linear filtering
- Means and offsets removal
- Re-sampling
- Physical units

## License

_Dymoval_ is licensed under
[BSD 3](https://github.com/ubaldot/dymoval/blob/main/LICENSE) license.
