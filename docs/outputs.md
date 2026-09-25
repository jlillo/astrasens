# Results and output files

```text
<root>/22_ANALYSIS/<night>/
├── DetectedSources/
├── Sensitivity/
└── Summary_plots/
```

PSF fit results include `x_fit`, `y_fit`, `fwhm_major`, `fwhm_minor`, `axis_ratio`, `amplitude`, and `delta_bic`. Diagnostics cover accepted and rejected candidates, rejection reasons (`reason`), fitting times, and optimizer evaluations.

Sensitivity files store separations, contrasts, recovery results, and metadata. A cached file may contain an unfinished run: the method and completion flag are checked before plotting.
