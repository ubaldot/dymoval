When picking up this package for maintenance, do the following:

- Run `zsh ./conda_dev_setup.sh` and then `conda activate dymoval_dev`. In
  this way you get an environment with all the greatest and latest
  dependencies,
- Run `pip install -e .` to have an editable install,
- Then, run `mypy ./src`
- Then, run `pytest -m "not (plots or open_tutorial)"`
- Then, try to commit: this trigger the git hooks and you will see quite
  many errors, To fix them, run `ruff format ./src` and `ruff check
  . --fix-only`
