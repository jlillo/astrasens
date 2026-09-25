# astrasens

**Detecting and characterizing companion sources in high-contrast astronomical images.**

astrasens is a Python tool for analyzing reduced or *drizzled* FITS images around a bright primary source. It combines primary-source modeling and subtraction, candidate generation, morphological validation through two-dimensional PSF fitting, and sensitivity estimation using artificial-source injections.

```{toctree}
:maxdepth: 2
:caption: User guide

installation
quickstart
tutorials
examples
method
outputs
limitations
```

## Workflow

1. Locate the primary source and estimate its core width.
2. Model and subtract the primary, retaining masks and valid pixels.
3. Generate candidate positions on the residual image.
4. Validate each candidate with an elliptical Moffat profile and a tilted background plane.
5. Remove invalid fits and duplicates, then write the catalog.
6. Inject artificial sources to estimate completeness during sensitivity analysis.

```{note}
Thresholds are heuristic until calibrated using data representative of the instrument, PSF sampling, and image-reconstruction procedure.
```
