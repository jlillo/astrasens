# Examples

## Stricter candidate selection

```bash
python astrasens_run.py image.fits data 240101 \
  --PSF-MIN-BIC 75 \
  --PSF-MAX-AXIS-RATIO 2.5 \
  --SENS-SEED 42
```

## Customize sensitivity analysis

`--SENSPAR` takes the number of separations, maximum magnitude contrast, contrast step, and number of injections:

```bash
python astrasens_run.py image.fits data 240101 \
  --SENSPAR 25 12 0.5 100 \
  --SENS-METHOD blind \
  --SENS-MATCH-RADIUS 1.0 -F
```

## Generate plots

```bash
python astrasens_run.py image.fits data 240101 --PLOTS
```

This mode reruns source detection and generates plots using an existing, complete sensitivity result.

The `reason` column in the diagnostics explains candidate rejections. Interpret missing catalog detections together with the sensitivity curve.
