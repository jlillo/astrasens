# How the code works

## Candidate generation and validation

The primary source is modeled and subtracted to produce a residual image. Candidate peaks are identified in this residual. This stage proposes positions; subsequent validation determines whether their morphology is consistent with a source.

Each candidate is fitted locally using a positive amplitude, a subpixel center, two FWHM values, an orientation, a Moffat `beta` parameter, and a background plane with gradients. The fit is compared with a plane-only model using `delta BIC`. Additional checks cover image edges, NaNs, masks, width, positional shifts, and axis ratio.

## Catalog generation

Nearby proposals are consolidated, retaining both their initial measurements and fit diagnostics. Proposal-stage measurements should not be interpreted as calibrated absolute PSF photometry.

## Sensitivity analysis

Artificial sources are injected at subpixel positions. Recovery requires a compatible detection within `--SENS-MATCH-RADIUS`. The `matched` method is a fast PSF-based proxy; `blind` uses the full detector.
