# astrasens

Determination of contrast (sensitivity) curves, detection of close companions, and diagnostic plots for high-spatial-resolution images obtained with the AstraLux-North (Calar Alto Observatory, Spain) and AstraLux-South instruments.

`astrasens` is intended for the detection and characterization of faint companion sources around bright targets in reduced or drizzled FITS images. It combines primary-source subtraction, candidate detection, two-dimensional PSF validation, and injection-based sensitivity calculations.

<img src="logo_astrasens.png" alt="alt text" width="500" />

Check out <a href="https://astrasens.readthedocs.io/en/latest/index.html">this page</a> for a complete documentation of the code.


## Usage

The code assumes that the data are organized using the following structure, where `root_path` is the main directory of the observation:

```
root_path/11_REDUCED/YYMMDD/
```

The code creates the following analysis folders when they do not already exist:

```
root_path/22_ANALYSIS/YYMMDD/DetectedSources/
root_path/22_ANALYSIS/YYMMDD/Sensitivity/
root_path/22_ANALYSIS/YYMMDD/Summary_plots/
```

The general way to run `astrasens` is:

```bash
python astrasens_run.py [file] [root_path] [YYMMDD]
```

For example:

```bash
python astrasens_run.py \
  TDRIZZLE_0100_TOI5377_SDSSz__240122.fits \
  /full_path/astrasens \
  240122
```

The input image is expected at:

```
/full_path/astrasens/11_REDUCED/240122/TDRIZZLE_0100_TOI5377_SDSSz__240122.fits
```

If an analysis product already exists, AstraSense can reuse it. Use `-FD` to force source detection again and `-F` to force recalculation of the sensitivity curve:

```bash
python astrasens_run.py image.fits /full_path/astrasens 240122 -FD -F
```

## Running a list of files

To run AstraSense over a list of files, create a plain ASCII file with one image name per row, for example `example.lis`, and run:

```bash
python astrasens_run.py example.lis /full_path/astrasens 240122
```

To run it over all matching images within a night:

```bash
python astrasens_run.py all /full_path/astrasens 240122
```

The `all` mode searches for files matching `TDRIZZLE*0100*.fits`. Successfully processed and failed files are recorded in `files_completed.lis` and `files_error.lis`.

## Target identification

If the target is a TESS Object of Interest, AstraSense can use the TIC identifier. For other targets, provide either the TIC identifier or the Gaia DR3 identifier when catalog information is required:

```bash
python astrasens_run.py \
  TDRIZZLE_0100_TARGET_SDSSz__240122.fits \
  /full_path/astrasens \
  240122 \
  --TIC 123456789
```

or:

```bash
python astrasens_run.py \
  TDRIZZLE_0100_TARGET_SDSSz__240122.fits \
  /full_path/astrasens \
  240122 \
  --GDR3 1040790426885870976
```

These options require access to the relevant external catalog services.

## Options

The most commonly used options are:

| Option | Description |
|---|---|
| `-G, --GDR3` | Gaia DR3 identifier of the observed source. |
| `-T, --TIC` | TIC identifier of the observed source. |
| `-V, --VERBOSE` | Print more information during the run. |
| `-P, --PLOTS` | Recreate plots from existing analysis products. |
| `-F, --FORCE` | Force recalculation of the sensitivity curve. |
| `-FD, --FORCEDET` | Force source detection again. |
| `-I, --IPAC` | Write detected-source output in IPAC format. |
| `-W, --WINDOW` | Maximum distance to consider. |
| `-PS, --pxscale` | Pixel scale in arcsec/pixel. Default: `0.02327`. |
| `--PSF-MIN-BIC` | Minimum improvement of the PSF model over a plane-only model. Default: `50`. |
| `--PSF-MAX-AXIS-RATIO` | Maximum fitted major/minor FWHM ratio. Default: `3`. |
| `--SENS-SEED` | Seed for reproducible injection angles. Default: `0`. |
| `--SENS-MATCH-RADIUS` | Maximum recovered/injected distance in pixels. Default: `1`. |
| `--SENS-METHOD` | Sensitivity method: `matched` or `blind`. Default: `matched`. |
| `-SP, --SENSPAR` | Sensitivity parameters: number of separations, maximum contrast, contrast step, and number of injected sources. |
| `-PA, --PARS` | Deprecated compatibility option. The current implementation uses a two-dimensional PSF fit. |

The complete command-line interface is available with:

```bash
python astrasens_run.py --help
```

## Sensitivity analysis

For example, to calculate a sensitivity curve using 25 separations, contrasts up to 12 magnitudes, a 0.5 magnitude step, and 100 injected sources per point:

```bash
python astrasens_run.py \
  image.fits \
  /full_path/astrasens \
  240122 \
  --SENSPAR 25 12 0.5 100 \
  --SENS-SEED 42 \
  --SENS-METHOD blind \
  -F
```

The `matched` method is a faster PSF-based recovery route. The `blind` method follows the full companion-detection path.

## What is produced

The detected-source products contain candidate positions and, when the PSF fit is accepted, fitted quantities such as:

```
x_fit, y_fit
fwhm_major, fwhm_minor
axis_ratio, amplitude
delta_bic
```

The PSF diagnostics include accepted and rejected candidates and the reason for each rejection. Sensitivity products contain the separation and contrast grids, recovery results, completeness information, and the configuration used for the calculation.

## Scientific considerations

The default `delta BIC = 50` criterion is a morphology heuristic and is not a calibrated false-alarm probability. Correlated pixels introduced by drizzling can affect the statistical interpretation of the detection criteria.

The sensitivity curve is conditional on the primary-source subtraction used in the run. A detected source should also be checked against image artifacts, background sources, astrometric consistency, and follow-up observations before it is interpreted as a physical companion.

## Documentation

The full documentation is available at <a href="https://astrasens.readthedocs.io/en/latest/index.html">this page</a>

## Examples

Locating companions:

![Residual image](https://github.com/jlillo/astrasens/blob/master/images/TOI5377_SDSSz__240122_0100__Residuals.png)

Determining the sensitivity curve:

![Sensitivity summary](https://github.com/jlillo/astrasens/blob/master/images/TOI5377_SDSSz__240122_0100__Summary.png)

Performing aperture photometry on detected companions:

![Aperture photometry](https://github.com/jlillo/astrasens/blob/master/images/TOI-1169_SDSSz__191029_0100__AperturePhot.png)

## Citing

So far, please use this publication (a JOSS publication is forthcoming):

```
@ARTICLE{2014A&A...566A.103L,
       author = {{Lillo-Box}, J. and {Barrado}, D. and {Bouy}, H.},
        title = "{High-resolution imaging of Kepler planet host candidates. A comprehensive comparison of different techniques}",
      journal = {\aap},
     keywords = {techniques: high angular resolution, planets and satellites: fundamental parameters, binaries: visual, Astrophysics - Earth and Planetary Astrophysics},
         year = 2014,
        month = jun,
       volume = {566},
          eid = {A103},
        pages = {A103},
          doi = {10.1051/0004-6361/201423497},
archivePrefix = {arXiv},
       eprint = {1405.3120},
 primaryClass = {astro-ph.EP},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2014A&A...566A.103L},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```


## License

`astrasens` is released under the MIT License. See the
[LICENSE](LICENSE) file for details.
