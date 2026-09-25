# Quick start

Place the image at `<root>/11_REDUCED/<night>/<image>.fits`:

```text
observations/
├── 11_REDUCED/240101/TDRIZZLE_0100_target_filter.fits
└── 22_ANALYSIS/          # created automatically
```

Run detection, sensitivity analysis, and plotting:

```bash
python astrasens_run.py TDRIZZLE_0100_target_filter.fits observations 240101
```

To force recalculation:

```bash
python astrasens_run.py TDRIZZLE_0100_target_filter.fits observations 240101 -F -FD
```

Key options:

```bash
python astrasens_run.py IMAGE ROOT NIGHT \
  --pxscale 0.02327 \
  --PSF-MIN-BIC 50 \
  --PSF-MAX-AXIS-RATIO 3 \
  --SENS-SEED 0 \
  --SENS-MATCH-RADIUS 1 \
  --SENS-METHOD matched
```

`--pxscale` is expressed in arcsec/pixel. The PSF limits control morphological validation. `--SENS-SEED` makes injection angles reproducible, and `--SENS-METHOD` accepts either `matched` or `blind`.
