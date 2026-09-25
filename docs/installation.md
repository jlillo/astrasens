# Installation

## From GitHub

Replace `OWNER/REPOSITORY` with the actual astrasens repository location:

```bash
git clone https://github.com/OWNER/REPOSITORY.git
cd REPOSITORY
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install numpy scipy astropy photutils matplotlib astroquery astroML emcee tqdm progressbar2 termcolor
```

The current implementation also imports `jlillo_pypref`. If this module belongs to a separate internal repository, install it before running astrasens.

## Check the installation

From the directory containing `astrasens_fitter.py`, `astrasens_run.py`, and `astrasens_plot.py`:

```bash
python -c "import numpy, scipy, astropy, photutils; print('Core dependencies: OK')"
python astrasens_run.py --help
```

```{warning}
The code expects input data under `11_REDUCED/<night>` and writes results under `22_ANALYSIS/<night>`. Check that your directory structure follows this convention.
```
