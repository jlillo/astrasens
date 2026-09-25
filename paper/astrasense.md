---
title: 'astrasens: Companion detection and sensitivity curves in high-spatial resolution astronomical images'
authors:
  - name: Jorge Lillo-Box
    orcid: '0000-0003-3742-1987'
    affiliation: 1
    corresponding: true
    email: Jorge.Lillo@cab.inta-csic.es
affiliations:
  - name: Centro de Astrobiología (CAB), CSIC-INTA, , Camino Bajo del Castillo s/n, 28692, Villanueva de la Cañada (Madrid), Spain
    index: 1
date: 2026-09-25
bibliography: astrasens.bib
repository: 'https://github.com/jlillo/astrasens'
---

# Summary

`astrasens` is an open-source Python package for detecting and characterising faint companion sources in high-contrast astronomical images. The code is optimized for its use in the AstraLux-North [@hormuth08] and AstraLux-South [@hippler09] lucky imaging instruments.It is designed to operate on images reduced by the official pipelines running at the corresponding observatories. Typically, a bright primary object dominates the image with a structured point-spread function (PSF). This, together with correlated photometric noise and nearby sources make ordinary point-source detection unreliable. `astrasens` combines primary-source modelling and subtraction, candidate generation on residual images, local image diagnostics, two-dimensional PSF fitting, catalogue generation, and injection-based sensitivity analysis in a reproducible workflow.

The software is intended for astronomers who need to search for close companions around bright targets, assess candidate reliability, and quantify detector performance. It accepts FITS images and produces candidate tables and diagnostic products that preserve fitted positions, widths, morphology, model-improvement statistics, rejection reasons, and sensitivity-trial outcomes.

# Statement of need

High-contrast imaging searches are often limited not by the availability of images, but by the difficulty of distinguishing a faint companion from structured residuals around a bright primary. A single global noise estimate and a fixed roundness or sharpness cut can be inadequate when the noise varies with separation, the PSF is asymmetric, or the image has been drizzled. These effects are particularly important for high-resolution optical observations that combine many short exposures and contain correlated residual structure.

`astrasens` addresses this need by making detection explicit and inspectable. Candidate proposals are separated from candidate validation: a broad proposal stage identifies possible peaks, while a local two-dimensional elliptical Moffat model plus a tilted background tests whether a candidate is better represented by a compact source than by the local background. The package records accepted and rejected fits, including failures caused by image boundaries, masks, parameter bounds, excessive axis ratio, or insufficient model improvement.

The package also provides a sensitivity curve based on the same detection logic used for the companion detection module. Artificial sources are injected at subpixel positions and are counted as recovered only when an astrometrically compatible detection is found and the position was not already occupied by a source in the un-injected residual. This design supports later calibration of completeness, false-positive rates, and the effects of primary-source subtraction.

# State of the field

Astronomical source detection is commonly implemented with matched filtering, thresholding, and PSF-fitting tools. `DAOStarFinder` in Photutils provides a widely used mechanism for detecting point-like sources in astronomical images [@photutils]. Astropy supplies the FITS, table, and modelling infrastructure on which many Python astronomy workflows depend [@astropy]. `astrasens` is complementary to these general-purpose components: it uses them for candidate proposal and scientific-data handling, while adding a workflow specialised for residual structure in high-contrast companion searches.

The package is not intended to replace established PSF-subtraction or high-contrast imaging pipelines. Its contribution is a transparent detector and validation layer that can be used after primary-source subtraction, with explicit morphology diagnostics and a sensitivity experiment tied to the production detection path. The statistical interpretation of local thresholds remains dependent on calibration data, the number of independent resolution elements, and correlations introduced by image reconstruction. `astrasens` therefore reports diagnostic quantities and provisional selection criteria rather than presenting a fixed threshold as a universal false-alarm probability [@mawet2014].

# Software design

`astrasens` is implemented in Python and operates on two-dimensional FITS images. Its main stages are: (1) estimate the primary position and core width; (2) construct and subtract a primary PSF model while retaining masks and valid-support information; (3) locate candidate positions from a filtered residual image; (4) fit each candidate with a positive elliptical Moffat profile and local tilted background; (5) reject fits for explicit reasons and merge duplicate positions; and (6) write source catalogues, PSF diagnostics, and optional injection-recovery results.

The companion model fits amplitude, subpixel centre, major and minor FWHM, orientation, Moffat beta, and background-plane parameters. The candidate table retains proposal-stage columns and adds fitted FWHMs, axis ratio, model-improvement statistic, and relative-width diagnostics. The implementation does not relabel proposal-stage fluxes as PSF photometry, avoiding conflation of detection measurements with calibrated photometry.

The sensitivity implementation uses the same primary subtraction, central mask, proposal limit, PSF validation, selection limits, and duplicate handling as the catalogue route. Trial status and metadata are saved incrementally, allowing interrupted experiments to resume. The current implementation measures completeness conditional on a fixed primary subtraction; repeating the primary fit for every injection is a separate calibration step because it measures an additional source of flux loss.

The repository includes offline tests for width calculations, half-maximum measurements, elliptical PSF fits, background gradients, masks and NaNs, image edges, duplicate handling, empty proposals, CLI defaults, and sensitivity bookkeeping. Diagnostic scripts and example outputs support reproduction on supplied FITS data. The package uses the established scientific Python stack, including NumPy, SciPy, Astropy, and Photutils.

# Research impact statement

`astrasens` provides a reproducible basis for companion searches in high-contrast images and for quantifying detector behaviour as a function of separation and contrast. Its immediate research value is methodological: it makes candidate validation, rejection decisions, and sensitivity trials available as inspectable data products. This supports studies of close companions, multiplicity, and faint sources around bright targets, while allowing users to recalibrate thresholds for their own detector, observing mode, PSF sampling, and image-reconstruction procedure. 

`astrasens` is particularly useful for programs aiming at exoplanet validation from transiting planet candidates detected by space-based photometers like *Kepler* [@borucki10], TESS [@ricker14] or PLATO [@rauer14] in the near future, through discarding close-in companions that may mimick the planetary transit signal. Within this context, this sensitivity curve can then be used by codes like the Background Source Confidence (`bsc`, [@lillo-box14]) to determine the probability of hidden undetected sources.


# AI usage disclosure

Generative AI was used to assist with software review, documentation, and preparation of this manuscript. The author verified the generated material against the source code, local tests, diagnostic outputs, and cited documentation. Scientific claims about performance and limitations are restricted to results recorded in the repository; the author remains responsible for the final manuscript, software, and interpretation.

# Acknowledgements

The author acknowledges the developers and maintainers of Python, NumPy, SciPy, Astropy, Photutils, and the open-source libraries used by astrasens. Funding information, observing-program acknowledgements, and data-provider acknowledgements should be added here before submission.

# References
