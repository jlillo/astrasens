# Limitations and best practices

- `delta BIC = 50` is a morphological heuristic, not a calibrated statistical significance.
- *Drizzling* introduces pixel correlations, making a naive interpretation based on independent pixels inappropriate.
- Sensitivity is conditional on primary-source subtraction. Refitting the primary for each injection would also measure flux losses introduced by that stage.
- Estimating the primary FWHM requires a bright, unsaturated source sufficiently far from image edges.
- Gaia, MAST, and SIMBAD queries rely on external services.
- A detection alone does not establish a physical companion: artifact checks and, where appropriate, astrometric follow-up are required.

Calibrate completeness and false-positive rates using data from the same instrument and observing mode before publishing scientific results.
