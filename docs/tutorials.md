# Tutorials

## Analyze one image

1. Place the image in `<root>/11_REDUCED/<night>/`.
2. Run the command in the quick-start guide.
3. Inspect the catalog, PSF diagnostics, and sensitivity curve under `<root>/22_ANALYSIS/<night>/`.
4. If you change selection criteria, use `-F` and, where appropriate, `-FD` to avoid reusing incompatible cached results.

## Process a night of observations

```bash
python astrasens_run.py all observations 240101
```

The `all` mode selects files matching `TDRIZZLE*0100*.fits`. It records successful and failed filenames in `files_completed.lis` and `files_error.lis`.

Alternatively, provide a list of image names:

```bash
python astrasens_run.py image_list.lis observations 240101
```

## Resume sensitivity analysis

Sensitivity analysis saves its progress in an `.npz` file. Repeat the same command to resume a compatible cache; an incompatible configuration requires recalculation with `-F`.

## Supply Gaia or TIC identifiers

`--GDR3` and `--TIC` accept catalog identifiers for external queries. These queries require access to the corresponding astronomical services.
