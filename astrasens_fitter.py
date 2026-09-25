import os
from sys import platform
if platform == 'linux2': plt.switch_backend('agg')
import argparse

import numpy as np
from numpy import unravel_index
from pylab import *
import time
import progressbar
from tqdm import tqdm
from termcolor import colored

from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.stats import SigmaClip
from astropy.visualization import SqrtStretch, simple_norm
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.table import Table, Column, MaskedColumn
from astropy.io import ascii
from astropy.coordinates import SkyCoord
import astropy.units as u

from photutils import datasets
from photutils import DAOStarFinder
from photutils import CircularAperture, CircularAnnulus
from photutils import find_peaks
from photutils.aperture import aperture_photometry, ApertureStats
from photutils.centroids import centroid_sources

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches
from matplotlib.colorbar import Colorbar
import matplotlib.gridspec as gridspec # GRIDSPEC !
from matplotlib.colors import LogNorm

from astroquery.mast import Catalogs
from astroquery.simbad import Simbad
import re

from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings("ignore") 

import astrasens_plot  as plotting
from astroML.stats import sigmaG


"""
	Automatic analysis of AstraLux images to get the sensitivity curve.

	===== SYNTAX =====
	date:	Night to be reduced: YYMMDD
	-SF	:	Setup file location can be modified as --SF path_to_file

	===== HISTORY =====
	2019/05/08		jlillobox		First version released

"""

# ===========================================================================================================
# 						ANCILLARY FUNCTIONS
# ===========================================================================================================

def psf_gauss(x, g0, g2, g3):
	"""
	PSF function to fit the radial profile of the target
	"""
	G = g0*np.exp(-x**2./(2.*g2**2.)) + g3

	return G

def psf_lorenz(x, g0, g1, l2):
	"""
	PSF function to fit the radial profile of the target
	"""
	L = g0 * 1./np.pi * 0.5*l2/((x-g1)**2. + (0.5*l2)**2)
	return L
def psf_lorenz2(x, g0, g1, l2, level,slope):
	"""
	PSF function to fit the radial profile of the target
	"""
	L = g0 * 1./np.pi * 0.5*l2/((x-g1)**2. + (0.5*l2)**2) + level + slope*x

	# z = (x-g1)/(0.5*l2)
	# L = g0 / (1+z**2) +level
	return L

def psf_func(x, g0, g1, g2, g3, l0, l2):
	"""
	PSF function to fit the radial profile of the target
	"""
	G = l0 * g0*np.exp(-(x-g1)**2/(2*g2**2)) + g3
	#L =   l2**2 / ((x-g1)**2 + l2**2) * l0/ np.pi

	L = g0 * 1./np.pi * 0.5*l2/((x-g1)**2 + (0.5*l2)**2)
	#M =  l0*g0*(1./(((x-g1)/l2)**2 +1.))**beta

	return G+L

def psf_func_alternative(x, g1, l0, l2):
	"""
	PSF function to fit the radial profile of the target with just a Gaussian
	"""
	L =   l2**2 / ((x-g1)**2 + l2**2) * l0/ np.pi

	return L

def get_synth_sigma(x, y, background=0.0):
    """Half the 15.865--84.135 percentile interval of a nonnegative 1D profile.

    This estimates sigma only for a well-sampled, untruncated Gaussian. It is
    neither the centroid nor the FWHM of a general PSF. Subtract the model's
    background explicitly; a negative profile is not a valid density.
    """
    from scipy.integrate import cumulative_trapezoid
    x = np.asarray(x, dtype=float)
    signal = np.asarray(y, dtype=float) - background
    if (x.ndim != 1 or signal.shape != x.shape or x.size < 3
            or not np.all(np.isfinite(x)) or not np.all(np.isfinite(signal))
            or np.any(np.diff(x) <= 0) or np.any(signal < 0)):
        raise ValueError('Expected increasing x and a finite, nonnegative profile.')
    cdf = cumulative_trapezoid(signal, x, initial=0)
    if cdf[-1] <= 0:
        raise ValueError('The background-subtracted profile has no positive area.')
    cdf /= cdf[-1]
    lo, hi = np.interp([0.158655253931457, 0.841344746068543], cdf, x)
    return 0.5 * (hi - lo)


def profile_fwhm(x, y, background=0.0):
    """Interpolated width between the nearest half-maximum crossings.

    Works for sampled Gaussian, Lorentzian, and mixed profiles without assuming
    a conversion from sigma. Raises if either crossing lies outside the input.
    """
    x = np.asarray(x, dtype=float)
    signal = np.asarray(y, dtype=float) - background
    if (x.ndim != 1 or signal.shape != x.shape or x.size < 3
            or not np.all(np.isfinite(x)) or not np.all(np.isfinite(signal))
            or np.any(np.diff(x) <= 0)):
        raise ValueError('Expected a finite profile sampled at increasing x.')
    peak = int(np.argmax(signal))
    half = signal[peak] / 2.0
    left = np.flatnonzero(signal[:peak] <= half)
    right = np.flatnonzero(signal[peak + 1:] <= half) + peak + 1
    if half <= 0 or not left.size or not right.size:
        raise ValueError('The profile does not bracket both half-maximum crossings.')
    i, j = left[-1], right[0]
    xl = x[i] + (half-signal[i]) * (x[i+1]-x[i]) / (signal[i+1]-signal[i])
    xr = x[j-1] + (half-signal[j-1]) * (x[j]-x[j-1]) / (signal[j]-signal[j-1])
    return float(xr-xl)


def target_core_fwhm(data, x, y, radius=20):
    """Observed core FWHM in pixels, from horizontal/vertical peak cuts.

    The outer fifth of each cut estimates its local pedestal (sky plus broad
    halo). This measures the sampled core, not the halo or a deconvolved PSF.
    A bright, unsaturated target is required. It only sets the proposal scale;
    companion widths are fitted independently.
    """
    data = np.asarray(data)
    xc, yc = int(round(float(x))), int(round(float(y)))
    ny, nx = data.shape
    if min(xc, yc, nx-1-xc, ny-1-yc) < radius:
        raise ValueError('Target is too close to an edge to measure its core.')
    profiles = (data[yc, xc-radius:xc+radius+1],
                data[yc-radius:yc+radius+1, xc])
    widths = []
    for profile in profiles:
        nedge = max(2, len(profile)//5)
        pedestal = np.median(np.r_[profile[:nedge], profile[-nedge:]])
        widths.append(profile_fwhm(np.arange(len(profile)), profile, pedestal))
    return float(np.sqrt(widths[0]*widths[1]))


def fit_companion_psf(data, x, y, initial_fwhm=3.0, fit_radius=25,
                      fwhm_bounds=(0.8, 40.0), max_axis_ratio=3.0,
                      max_shift=5.0, min_delta_bic=50.0):
    """Validate morphology with an elliptical Moffat plus a tilted background.

    Fits positive amplitude, subpixel centre, two FWHMs, angle, beta and a
    background plane. Widths need not equal the primary's narrow core: lucky
    imaging companions can have a much less prominent core. Delta BIC compares
    this model with a plane alone. It is a morphology heuristic, NOT a calibrated
    detection significance (in particular for correlated drizzle pixels). The
    conservative default of 50 needs calibration on a representative dataset.

    Returns diagnostics for both accepted and rejected candidates. No hidden
    clipping of data, replacement of NaNs, or DAO roundness veto is performed.
    """
    import time
    started = time.perf_counter()
    from scipy.optimize import least_squares
    if (fit_radius < 4 or fwhm_bounds[0] <= 0
            or fwhm_bounds[1] <= fwhm_bounds[0] or max_shift <= 0
            or max_axis_ratio < 1 or not np.isfinite(initial_fwhm)
            or initial_fwhm <= 0):
        raise ValueError('Invalid PSF fitting options.')
    result = dict(x_input=float(x), y_input=float(y), x_fit=np.nan, y_fit=np.nan,
                  fwhm_major=np.nan, fwhm_minor=np.nan, axis_ratio=np.nan,
                  amplitude=np.nan, delta_bic=np.nan, width_relerr=np.nan,
                  accepted=False, reason='invalid_position', duplicate_of=-1,
                  fit_seconds=0.0, optimizer_starts=0, optimizer_nfev=0,
                  model_evaluations=0)

    def finish():
        result['fit_seconds'] = time.perf_counter()-started
        return result
    if not np.isfinite(x) or not np.isfinite(y):
        return finish()
    xc, yc = int(round(x)), int(round(y))
    ny, nx = data.shape
    r = int(fit_radius)
    if min(xc, yc, nx-1-xc, ny-1-yc) < r:
        result['reason'] = 'edge'
        return finish()
    stamp = np.asarray(data[yc-r:yc+r+1, xc-r:xc+r+1], dtype=float)
    yy, xx = np.mgrid[-r:r+1, -r:r+1]
    valid = np.isfinite(stamp)
    if valid.mean() < 0.9 or not valid[r, r]:
        result['reason'] = 'masked_stamp'
        return finish()
    xx, yy, z = xx[valid].astype(float), yy[valid].astype(float), stamp[valid]
    plane = np.column_stack((np.ones(z.size), xx, yy))
    edge = np.hypot(xx, yy) >= 0.8*r
    background = np.linalg.lstsq(plane[edge], z[edge], rcond=None)[0]
    noise = float(sigma_clipped_stats((z-plane@background)[edge])[2])
    if not np.isfinite(noise) or noise <= 0:
        result['reason'] = 'invalid_noise'
        return finish()
    null_residual = z-plane@np.linalg.lstsq(plane, z, rcond=None)[0]
    rss_null = float(null_residual@null_residual)
    amplitude = max(float(np.max(z-plane@background)), noise)

    def model(p):
        amp, dx, dy, fw1, fw2, angle, beta, b, bx, by = p
        u = (xx-dx)*np.cos(angle)+(yy-dy)*np.sin(angle)
        v = -(xx-dx)*np.sin(angle)+(yy-dy)*np.cos(angle)
        rho = (u/fw1)**2+(v/fw2)**2
        return amp*(1+4*np.expm1(np.log(2)/beta)*rho)**(-beta)+b+bx*xx+by*yy

    def residual(p):
        result['model_evaluations'] += 1
        return (model(p)-z)/noise

    lower = [0, -max_shift, -max_shift, *([fwhm_bounds[0]]*2),
             -np.pi/2, 1.1, -np.inf, -np.inf, -np.inf]
    upper = [np.inf, max_shift, max_shift, *([fwhm_bounds[1]]*2),
             np.pi/2, 6, np.inf, np.inf, np.inf]
    best = None
    # Multiple starting widths avoid selecting only the primary's narrow core.
    starts = np.unique(np.clip([initial_fwhm, 3*initial_fwhm, 0.5*fwhm_bounds[1]],
                               1.01*fwhm_bounds[0], 0.99*fwhm_bounds[1]))
    for width in starts:
        p0 = [amplitude, 0, 0, width, width, 0, 2, *background]
        try:
            result['optimizer_starts'] += 1
            fit = least_squares(residual, p0,
                                bounds=(lower, upper), max_nfev=200)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue
        result['optimizer_nfev'] += int(fit.nfev)
        if fit.success and np.all(np.isfinite(fit.x)):
            if best is None or np.sum(fit.fun**2) < np.sum(best.fun**2):
                best = fit
    if best is None:
        result['reason'] = 'fit_failed'
        return finish()
    p = best.x
    rss = float(np.sum((model(p)-z)**2))
    major, minor = max(p[3:5]), min(p[3:5])
    # 10 parameters for Moffat+plane versus 3 for the plane. The unknown variance
    # term cancels. This comparison does not assume the noise estimate is exact.
    tiny = np.finfo(float).tiny
    delta_bic = z.size*np.log(max(rss_null, tiny)/max(rss, tiny))-7*np.log(z.size)
    width_relerr = np.inf
    try:
        if np.linalg.matrix_rank(best.jac) == len(p):
            covariance = np.linalg.inv(best.jac.T@best.jac)*np.sum(best.fun**2)/(z.size-len(p))
            width_relerr = float(max(np.sqrt(np.maximum(np.diag(covariance)[3:5], 0))/p[3:5]))
    except np.linalg.LinAlgError:
        pass
    result.update(x_fit=float(xc+p[1]), y_fit=float(yc+p[2]),
                  fwhm_major=float(major), fwhm_minor=float(minor),
                  axis_ratio=float(major/minor), amplitude=float(p[0]),
                  delta_bic=float(delta_bic), width_relerr=width_relerr)
    if max(abs(p[1]), abs(p[2])) >= 0.98*max_shift:
        result['reason'] = 'centroid_at_bound'
    elif minor <= 1.02*fwhm_bounds[0] or major >= 0.98*fwhm_bounds[1]:
        result['reason'] = 'width_at_bound'
    elif major/minor > max_axis_ratio:
        result['reason'] = 'elongated'
    elif delta_bic < min_delta_bic:
        result['reason'] = 'no_psf_improvement'
    else:
        result.update(accepted=True, reason='accepted')
    return finish()


def _companion_sources(data, XYcoords, fwhm, signif, fluxmin, VERBOSE,
                       psf_options=None, diagnostics=None):
    """DAO proposals with permissive morphology, then independent 2D validation."""
    import inspect
    import builtins
    import time
    started = time.perf_counter()
    _, median, std = sigma_clipped_stats(data, sigma=3.0)
    if not np.isfinite(std) or std <= 0:
        return None
    if XYcoords is not None and len(XYcoords) == 0:
        return None
    kwargs = dict(fwhm=float(fwhm), threshold=signif*abs(std), xycoords=XYcoords)
    # Photutils 3 renamed these arguments; support the installed 1.x API too.
    parameters = inspect.signature(DAOStarFinder).parameters
    if 'sharpness_range' in parameters:
        kwargs.update(sharpness_range=(-np.inf, np.inf), roundness_range=(-np.inf, np.inf))
    else:
        kwargs.update(sharplo=-np.inf, sharphi=np.inf, roundlo=-np.inf, roundhi=np.inf)
    proposed = DAOStarFinder(**kwargs).find_stars(data-median)
    if proposed is None or len(proposed) == 0:
        return None
    ny, nx = data.shape
    fitted_rows, fits = [], []
    print('PSF: {} DAO proposals; validating morphology...'.format(len(proposed)), flush=True)
    for i, source in tqdm(enumerate(proposed), total=len(proposed)):
        x, y = float(source['xcentroid']), float(source['ycentroid'])
        if not (0.05*nx < x < 0.95*nx and 0.05*ny < y < 0.95*ny
                and source['peak'] > fluxmin):
            continue
        result = fit_companion_psf(data, x, y, initial_fwhm=fwhm, **(psf_options or {}))
        result['dao_id'] = int(source['id'])
        fits.append(result)
        if result['accepted']:
            fitted_rows.append((i, result))
    # Merge multiple pixel maxima that converge to the same fitted source.
    # Rank by model evidence, not by the height of a possibly noisy pixel.
    fitted_rows.sort(key=lambda item: item[1]['delta_bic'], reverse=True)
    kept = []
    merge_radius = max(2.0, float(fwhm))
    for i, result in fitted_rows:
        # pylab's wildcard import replaces any with numpy.any, which treats a
        # generator as truthy even when empty. Use the explicit Python builtin
        # and record the actual retained source causing a duplicate rejection.
        duplicate = builtins.next((other for _, other in kept
            if np.hypot(result['x_fit']-other['x_fit'],
                        result['y_fit']-other['y_fit']) < merge_radius), None)
        if duplicate is not None:
            result.update(accepted=False, reason='duplicate',
                          duplicate_of=duplicate['dao_id'])
        else:
            kept.append((i, result))
    if diagnostics is not None:
        diagnostics.extend(fits)
    from collections import Counter
    print('PSF: {} fits, {} optimizer starts, {} model evaluations, {:.1f} s; {} kept'.format(
        len(fits), builtins.sum(f['optimizer_starts'] for f in fits),
        builtins.sum(f['model_evaluations'] for f in fits),
        time.perf_counter()-started, len(kept)), flush=True)
    print('2D PSF validation:', dict(Counter(fit['reason'] for fit in fits)), flush=True)
    if not kept:
        return None
    sources = proposed[[i for i, _ in kept]].copy()
    sources['xcentroid'] = [result['x_fit'] for _, result in kept]
    sources['ycentroid'] = [result['y_fit'] for _, result in kept]
    for key in ('fwhm_major', 'fwhm_minor', 'axis_ratio', 'delta_bic', 'width_relerr'):
        sources['psf_'+key] = [result[key] for _, result in kept]
    if VERBOSE:
        plt.imshow(data, origin='lower', norm=ImageNormalize(stretch=SqrtStretch()))
        plt.scatter(sources['xcentroid'], sources['ycentroid'], marker='x', c='gold')
        plt.title('Companions passing 2D PSF validation')
        plt.show()
        plt.close()
    return sources


def find_sources(data, XYcoords=None, fwhm=10., min_sharpness=0.8, roundness=0.1,
					signif=5.0, fluxmin=1., target_fwhm=None,
					SENS=False, VERBOSE=False, COMPANIONS=False, p0_target=None,
					psf_options=None, diagnostics=None):
	"""Find primary/sensitivity sources or validate companion PSFs in 2D.

	For COMPANIONS, roundness/min_sharpness and p0_target are retained only for
	call compatibility. Morphology is controlled by fit_companion_psf options;
	target_fwhm is the measured core FWHM in pixels, used as an initial scale.
	diagnostics, if supplied, is a list populated with PSF acceptance/rejection
	records. SENS is a legacy DAO-only option; the sensitivity pipeline now calls
	detect_companions instead. Primary DAO selection is unchanged.
	"""
	if COMPANIONS:
		core_fwhm = fwhm if target_fwhm is None else target_fwhm
		return _companion_sources(data, XYcoords, core_fwhm, signif, fluxmin,
									VERBOSE, psf_options, diagnostics)
	_, median, std = sigma_clipped_stats(data, sigma=3.0)
	daofind = DAOStarFinder(fwhm=float(fwhm), threshold=signif*abs(std),
							xycoords=XYcoords)
	candidates = daofind.find_stars(data-median)
	if candidates is None or len(candidates) == 0:
		return None
	ny, nx = data.shape
	keep = ((candidates['xcentroid'] > 0.05*nx)
			& (candidates['xcentroid'] < 0.95*nx)
			& (candidates['ycentroid'] > 0.05*ny)
			& (candidates['ycentroid'] < 0.95*ny))
	if not SENS:
		keep &= candidates['peak'] > fluxmin
	sources = candidates[keep]
	if len(sources) == 0:
		return None
	if VERBOSE:
		print("Find sources results: ")
		print(sources)
		plt.imshow(data, origin='lower', norm=ImageNormalize(stretch=SqrtStretch()))
		plt.scatter(sources['xcentroid'], sources['ycentroid'], marker='x', c='gold')
		plt.show()
		plt.close()
	return sources


def companion_psf_options(args):
    """One configuration for real companions and injection recovery."""
    return dict(min_delta_bic=float(getattr(args, 'PSF_MIN_BIC', 50.)),
                max_axis_ratio=float(getattr(args, 'PSF_MAX_AXIS_RATIO', 3.)))


def companion_residuals(data, fake, target_xy):
    """Subtract the fixed primary model and apply the same central mask."""
    residuals = np.asarray(data, dtype=float)-np.asarray(fake, dtype=float)
    x, y = target_xy
    ny, nx = residuals.shape
    residuals[max(0, int(y-5)):min(ny, int(y+5)),
              max(0, int(x-5)):min(nx, int(x+5))] = np.nan
    return residuals


def detect_companions(residuals, core_fwhm, args, diagnostics=None, verbose=False):
    """Shared blind search: same peak cap, DAO proposals, PSF fits and deduplication.

    Do not pass injection coordinates here: recovery must find its own peaks.
    The caller prepares the image with companion_residuals, including its mask.
    """
    peaks = findpeaks(residuals, npeaks=1000)
    coords = () if peaks is None else tuple(zip(peaks['x_peak'], peaks['y_peak']))
    return find_sources(residuals, XYcoords=coords, fwhm=core_fwhm,
                        fluxmin=2., signif=2., COMPANIONS=True,
                        target_fwhm=core_fwhm, VERBOSE=verbose,
                        psf_options=companion_psf_options(args), diagnostics=diagnostics)


def artificial_companion(shape, popt, x, y, dmag):
    """Translate/scale the existing analytic primary profile, excluding sky.

    Array coordinates are exact 0..N-1, with arbitrary subpixel injection phases.
    This retains the existing analytic PSF assumption and adds no extra noise.
    """
    p = np.asarray(popt, dtype=float)
    if p.shape != (6,) or not np.all(np.isfinite(p)) or p[2] <= 0 or p[5] <= 0:
        raise ValueError('Injection requires six finite PSF parameters with positive widths.')
    yy, xx = np.indices(shape, dtype=float)
    radius = np.hypot(xx-x, yy-y)
    signal = psf_func(radius, *p)-p[3]
    if not np.all(np.isfinite(signal)) or np.min(signal) < -1.e-10:
        raise ValueError('The primary model cannot define a nonnegative artificial PSF.')
    return np.maximum(signal, 0.) * 10.**(-float(dmag)/2.5)


def _array_digest(array):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float64).tobytes()).hexdigest()


def sensitivity_setup(data, popt, fake, target_xy, args):
    """Grids and complete checkpoint identity. WINDOW limits separations, not pixels."""
    import json
    import hashlib
    ndist, maxmag, step, nstars = map(float, args.SENSPAR)
    if (not np.all(np.isfinite([ndist, maxmag, step, nstars])) or ndist < 1
            or nstars < 1 or int(ndist) != ndist or int(nstars) != nstars
            or maxmag <= 0 or step <= 0 or args.pxscale <= 0):
        raise ValueError('Invalid SENSPAR or pixel scale.')
    seed = int(getattr(args, 'SENS_SEED', 0))
    match_radius = float(getattr(args, 'SENS_MATCH_RADIUS', 1.0))
    if seed < 0 or not np.isfinite(match_radius) or match_radius <= 0:
        raise ValueError('Seed must be nonnegative and matching radius positive.')
    xt, yt = map(float, target_xy)
    ny, nx = data.shape
    if np.shape(fake) != data.shape:
        raise ValueError('Primary model and science image must have the same shape.')
    # Keep injection centres in the normal fit/border footprint. Never crop the
    # detection image: cropping would change the noise, peak cap and border cuts.
    mx, my = max(25., .05*nx), max(25., .05*ny)
    maxdist = min(xt-mx, nx-1-mx-xt, yt-my, ny-1-my-yt)*args.pxscale
    window = getattr(args, 'WINDOW', None)
    if window is not None and (not np.isfinite(window) or window <= 0):
        raise ValueError('WINDOW must be positive.')
    maxdist = min(maxdist, float(window)/2 if window is not None else 6.)
    if maxdist < .1:
        raise ValueError('Insufficient valid field for separations starting at 0.1 arcsec.')
    dist_arr = np.geomspace(.1, maxdist, int(ndist))
    dmag_arr = np.r_[np.arange(maxmag, 0., -step), 0.]
    peaks = findpeaks(data, npeaks=1)
    if peaks is None:
        raise ValueError('Cannot measure the primary for sensitivity trials.')
    core = target_core_fwhm(data, peaks['x_peak'][0], peaks['y_peak'][0])
    settings = dict(method='shared-companion-psf-v1', recovery_method=getattr(args, 'SENS_METHOD', 'matched'), recovery_version='v5-local-control-peaks', subtraction='fixed-primary-residual',
                    psf_options=companion_psf_options(args), core_fwhm=core,
                    npeaks=1000, fluxmin=2., signif=2., central_mask_half_size=5,
                    primary_xy=[xt, yt], shape=list(data.shape), popt=list(map(float, popt)),
                    pxscale=float(args.pxscale), window=window, seed=seed,
                    match_radius=match_radius, nstars=int(nstars),
                    dist_arr=dist_arr.tolist(), dmag_arr=dmag_arr.tolist(),
                    data_sha256=_array_digest(data), model_sha256=_array_digest(fake),
                    exclusion='baseline half-major-FWHM or matching radius; conditional on clean locations')
    metadata = json.dumps(settings, sort_keys=True)
    key = hashlib.sha256(metadata.encode()).hexdigest()
    return dist_arr, dmag_arr, core, metadata, key


def _save_sensitivity_checkpoint(path, payload):
    """Atomic update after each trial; a partial curve is never called complete."""
    import os
    import tempfile
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.', suffix='.npz', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as output:
            np.savez(output, **payload)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def completeness_limits(dmag_arr, detection, level=.95):
    """First loss of completeness moving from bright to faint injections.

    Require the bright end to pass. A disconnected faint recovery island must
    not create a sensitivity limit. Missing trials at the crossing remain NaN.
    The faintest sampled contrast is a grid bound, not a measured crossing.
    """
    mags=np.asarray(dmag_arr,dtype=float)
    rates=np.atleast_2d(detection)
    order=np.argsort(mags)
    limits=np.full(len(rates),np.nan)
    if not len(order):
        return limits
    for row,fraction in enumerate(rates):
        if not np.isfinite(fraction[order[0]]) or fraction[order[0]]<level:
            continue
        for j in range(1,len(order)):
            bright,faint=order[j-1],order[j]
            if not np.isfinite(fraction[faint]):
                break
            if fraction[faint]<level:
                limits[row]=np.interp(level,[fraction[faint],fraction[bright]],
                                     [mags[faint],mags[bright]])
                break
        else:
            limits[row]=mags[order[-1]]
    return limits


def compute_sensitivity(data, popt, fake, target_xy, args, checkpoint=None, restart=False):
    """Injection completeness using the exact blind companion detector.

    The primary subtraction/core scale are held fixed, as in the previous
    residual-based experiment. This is NOT an end-to-end refit/self-subtraction
    calibration. Pre-existing detections are excluded and reported separately.
    """
    import json
    import time
    from pathlib import Path
    dist_arr, dmag_arr, core, metadata, key = sensitivity_setup(data, popt, fake, target_xy, args)
    settings = json.loads(metadata)
    shape = (len(dist_arr), len(dmag_arr), settings['nstars'])
    state = np.full(shape, -1, dtype=np.int8)  # pending=-1, missed=0, recovered=1, excluded=2
    matched_distance = np.full(shape, np.nan)
    trial_seconds = np.zeros(shape)
    angles = np.random.default_rng(settings['seed']).uniform(0, 2*np.pi, size=shape)
    residuals = companion_residuals(data, fake, target_xy)
    baseline_xy, baseline_radius = None, None
    if checkpoint is not None and Path(checkpoint).exists() and not restart:
        with np.load(checkpoint, allow_pickle=False) as saved:
            if 'cache_key' not in saved or str(saved['cache_key'].item()) != key:
                raise ValueError('Incompatible sensitivity cache: use -F to recalculate.')
            state = saved['trial_status'].copy()
            matched_distance = saved['matched_distance'].copy()
            trial_seconds = saved['trial_seconds'].copy()
            angles = saved['angles'].copy()
            baseline_xy = saved['baseline_xy'].copy()
            baseline_radius = saved['baseline_radius'].copy()
            if state.shape != shape:
                raise ValueError('Invalid checkpoint trial shape.')
    if baseline_xy is None:
        print('Sensitivity: baseline search with the shared companion detector.', flush=True)
        baseline = detect_companions(residuals, core, args)
        baseline_xy = np.empty((0, 2))
        baseline_radius = np.empty(0)
        if baseline is not None and len(baseline):
            baseline_xy = np.column_stack((baseline['xcentroid'], baseline['ycentroid']))
            baseline_radius = np.maximum(settings['match_radius'], .5*np.asarray(baseline['psf_fwhm_major']))

    def snapshot():
        valid = (state == 0) | (state == 1)
        attempted = np.sum(valid, axis=2)
        recovered = np.sum(state == 1, axis=2)
        excluded = np.sum(state == 2, axis=2)
        detection = np.divide(recovered, attempted, out=np.full(attempted.shape, np.nan), where=attempted > 0)
        payload = dict(detection=detection, dist_arr=dist_arr, dmag_arr=dmag_arr,
                       n_recovered=recovered, n_trials=attempted, n_excluded=excluded,
                       trial_status=state, matched_distance=matched_distance,
                       trial_seconds=trial_seconds, angles=angles, baseline_xy=baseline_xy,
                       baseline_radius=baseline_radius, metadata=metadata, cache_key=key,
                       method='shared-companion-psf-v1', complete=bool(np.all(state != -1)),
                       completeness_level=.95, contrast_limit=completeness_limits(dmag_arr, detection))
        if checkpoint is not None:
            _save_sensitivity_checkpoint(checkpoint, payload)
        return payload

    result = snapshot()
    pending = np.argwhere(state == -1)
    print('Sensitivity: {} trials pending ({} total); one full blind search per injection.'.format(
        len(pending), state.size), flush=True)
    xt, yt = target_xy
    for number, (i, j, k) in enumerate(pending, 1):
        started = time.perf_counter()
        theta = angles[i, j, k]
        x = xt+dist_arr[i]/args.pxscale*np.cos(theta)
        y = yt+dist_arr[i]/args.pxscale*np.sin(theta)
        if np.any(np.hypot(baseline_xy[:, 0]-x, baseline_xy[:, 1]-y) <= baseline_radius):
            state[i, j, k] = 2
        else:
            injected = artificial_companion(data.shape, popt, x, y, dmag_arr[j])
            # Inject before the shared masking step, so a source hidden under the
            # central mask is counted as missed rather than restored through it.
            trial = companion_residuals(data+injected, fake, target_xy)
            found = detect_companions(trial, core, args)
            state[i, j, k] = 0
            if found is not None and len(found):
                positions = np.column_stack((found['xcentroid'], found['ycentroid']))
                distances = np.hypot(positions[:, 0]-x, positions[:, 1]-y)
                # An old detection cannot be counted as a newly recovered source.
                old = np.zeros(len(positions), dtype=bool)
                for pos, radius in zip(baseline_xy, baseline_radius):
                    old |= np.hypot(positions[:, 0]-pos[0], positions[:, 1]-pos[1]) <= radius
                if np.any(~old):
                    nearest = float(np.min(distances[~old]))
                    matched_distance[i, j, k] = nearest
                    state[i, j, k] = int(nearest < settings['match_radius'])
        trial_seconds[i, j, k] = time.perf_counter()-started
        result = snapshot()
        print('Sensitivity {}/{}: r={:.3f} arcsec, contrast={:.2f}, status={}, {:.1f} s (saved)'.format(
            number, len(pending), dist_arr[i], dmag_arr[j], int(state[i, j, k]),
            trial_seconds[i, j, k]), flush=True)
    return result


def matched_injection_score(residual, popt, x, y, dmag, core_fwhm, match_radius=1.):
    """Local peak recovery with an identical un-injected control.

    This fast detector searches a PSF-filtered patch for peaks; it does not force
    a fit at the injection position. It is still a proxy, not the full blind
    Moffat companion detector. Noise is measured in the filtered control image,
    including pixel correlations. Both a significant peak and a significant
    increment are required; a control peak at the same position is rejected.
    The threshold of 5 is an empirical local-noise ratio, not a calibrated
    false-alarm probability in these correlated, structured residuals.
    """
    from scipy.signal import fftconvolve
    from scipy.ndimage import maximum_filter
    result = dict(score=np.nan, recovered=False, reason='invalid_position',
                  valid_energy_fraction=0., noise=np.nan, baseline_score=np.nan,
                  increment_score=np.nan, matched_distance=np.nan)
    if not np.all(np.isfinite([x,y,dmag,core_fwhm,match_radius])) or min(core_fwhm,match_radius)<=0:
        return result
    ny,nx=residual.shape
    xc,yc=int(round(x)),int(round(y))
    if not (0<=xc<nx and 0<=yc<ny):
        result['reason']='outside_image'; return result
    if not np.isfinite(residual[yc,xc]):
        result['reason']='masked_centre'; return result
    # Keep the filter compact: the broad primary halo is not a detection core.
    kr=max(3,int(np.ceil(2*core_fwhm)))
    kernel=artificial_companion((2*kr+1,2*kr+1),popt,kr,kr,0.)
    kernel=kernel-np.mean(kernel)
    kernel/=np.sqrt(np.sum(kernel**2))
    width=max(float(core_fwhm),abs(float(popt[2])),abs(float(popt[5])))
    radius=max(int(np.ceil(3*width))+kr,4*kr)
    xlo,xhi=max(0,xc-radius),min(nx,xc+radius+1)
    ylo,yhi=max(0,yc-radius),min(ny,yc+radius+1)
    patch=np.asarray(residual[ylo:yhi,xlo:xhi],dtype=float)
    yy,xx=np.indices(patch.shape,dtype=float)
    rho=np.hypot(xx+xlo-x,yy+ylo-y)
    valid=np.isfinite(patch)
    signal=artificial_companion(patch.shape,popt,x-xlo,y-ylo,dmag)
    # Every local filter fits an intercept on its finite pixels. Missing pixels
    # are excluded from numerator AND template normalization, never unmasked.
    ones=np.ones(kernel.shape)
    conv=lambda a,k: fftconvolve(a,k[::-1,::-1],mode='same')
    count=conv(valid.astype(float),ones)
    ksum=conv(valid.astype(float),kernel)
    energy=conv(valid.astype(float),kernel**2)
    norm=np.sqrt(np.maximum(energy-ksum**2/np.maximum(count,1),0))
    support=(count>=.8*kernel.size)&(energy>=.5)&(norm>0)&valid
    def response(image):
        values=np.where(valid,image,0.)
        raw=conv(values,kernel)-conv(values,ones)*ksum/np.maximum(count,1)
        return np.divide(raw,norm,out=np.full(patch.shape,np.nan),where=support)
    before=response(patch)
    addition=response(signal)
    after=before+addition
    sky=support&(rho>max(1.5*width,2.5*kr))&(rho<max(2.5*width,3.5*kr))
    if np.count_nonzero(sky)<20:
        result['reason']='insufficient_sky'; return result
    background=float(np.median(before[sky]))
    noise=float(np.std(before[sky]))
    result['noise']=noise
    if not np.isfinite(noise) or noise<=0:
        result['reason']='invalid_noise'; return result
    baseline=(before-background)/noise
    injected=(after-background)/noise
    increment=addition/noise
    centre=(yc-ylo,xc-xlo)
    result.update(score=float(injected[centre]),baseline_score=float(baseline[centre]),
                  increment_score=float(increment[centre]),
                  valid_energy_fraction=float(np.clip(energy[centre],0,1)))
    if not support[centre]:
        result['reason']='insufficient_psf_support'; return result
    def peaks(scores):
        values=np.where(support,scores,-np.inf)
        # Search beyond the match circle, so a slope toward the primary cannot
        # masquerade as a peak at the boundary of the injection matching region.
        return np.argwhere(support&(rho<=2*core_fwhm+match_radius)&(scores>=5.)
                           &(values==maximum_filter(values,size=3,mode='constant',cval=-np.inf)))
    candidates=peaks(injected)
    control=peaks(baseline)
    if not len(candidates):
        result['reason']='no_peak'; return result
    distances=np.hypot(candidates[:,1]+xlo-x,candidates[:,0]+ylo-y)
    order=np.argsort(distances)
    for j in order:
        if distances[j]>match_radius:
            continue
        py,px=candidates[j]
        result.update(score=float(injected[py,px]),baseline_score=float(baseline[py,px]),
                      increment_score=float(increment[py,px]),matched_distance=float(distances[j]))
        if len(control) and np.any(np.hypot(control[:,1]-px,control[:,0]-py)<=max(match_radius,.5*core_fwhm)):
            result['reason']='control_peak'; continue
        if increment[py,px]<5.:
            result['reason']='insufficient_increment'; continue
        result.update(recovered=True,reason='recovered'); return result
    if result['reason']=='invalid_position':
        result['reason']='peak_not_matched'
    return result


def compute_sensitivity_fast(data, popt, fake, target_xy, args, checkpoint=None, restart=False):
    """Local peak completeness with matched-PSF injections and null controls.

    The science residual, central mask and analytic injection model are shared
    with the reference experiment. This remains a proxy for blind PSF detection.
    """
    import json
    import time
    from pathlib import Path
    dist_arr, dmag_arr, core, metadata, key = sensitivity_setup(data, popt, fake, target_xy, args)
    settings = json.loads(metadata)
    shape = (len(dist_arr), len(dmag_arr), settings['nstars'])
    state = np.full(shape, -1, dtype=np.int8)
    scores = np.full(shape, np.nan)
    reasons = np.full(shape, 'pending', dtype='U32')
    coverage = np.zeros(shape)
    baseline_scores = np.full(shape, np.nan)
    increment_scores = np.full(shape, np.nan)
    matched_distances = np.full(shape, np.nan)
    seconds = np.zeros(shape)
    angles = np.random.default_rng(settings['seed']).uniform(0, 2*np.pi, size=shape)
    residual = companion_residuals(data, fake, target_xy)
    baseline_xy, baseline_radius = None, None
    if checkpoint is not None and Path(checkpoint).exists() and not restart:
        with np.load(checkpoint, allow_pickle=False) as saved:
            if ('cache_key' not in saved or str(saved['cache_key'].item()) != key
                    or str(saved['method'].item()) != 'matched-psf-completeness-v5'):
                raise ValueError('Incompatible sensitivity cache: use -F to recalculate.')
            state = saved['trial_status'].copy()
            scores = saved['matched_score'].copy()
            reasons = saved['trial_reason'].copy()
            coverage = saved['valid_energy_fraction'].copy()
            baseline_scores = saved['baseline_score'].copy()
            increment_scores = saved['increment_score'].copy()
            matched_distances = saved['matched_distance'].copy()
            seconds = saved['trial_seconds'].copy()
            angles = saved['angles'].copy()
            baseline_xy = saved['baseline_xy'].copy()
            baseline_radius = saved['baseline_radius'].copy()
            if state.shape != shape:
                raise ValueError('Invalid checkpoint trial shape.')
    # An empty, already-computed catalogue is a valid baseline. Resuming must not
    # redo the expensive blind search merely because no companions were found.
    if baseline_xy is None:
        baseline = detect_companions(residual, core, args)
        baseline_xy = np.empty((0, 2))
        baseline_radius = np.empty(0)
        if baseline is not None and len(baseline):
            baseline_xy = np.column_stack((baseline['xcentroid'], baseline['ycentroid']))
            baseline_radius = np.maximum(settings['match_radius'], .5*np.asarray(baseline['psf_fwhm_major']))

    def snapshot():
        trials = np.sum((state == 0) | (state == 1), axis=2)
        recovered = np.sum(state == 1, axis=2)
        excluded = np.sum(state == 2, axis=2)
        detection = np.divide(recovered, trials, out=np.full(trials.shape, np.nan), where=trials > 0)
        payload = dict(detection=detection, dist_arr=dist_arr, dmag_arr=dmag_arr,
                       n_recovered=recovered, n_trials=trials, n_excluded=excluded,
                       trial_status=state, matched_score=scores, trial_reason=reasons,
                       valid_energy_fraction=coverage, baseline_score=baseline_scores,
                       increment_score=increment_scores, matched_distance=matched_distances,
                       trial_seconds=seconds, angles=angles,
                       baseline_xy=baseline_xy, baseline_radius=baseline_radius,
                       metadata=metadata, cache_key=key, method='matched-psf-completeness-v5',
                       complete=bool(np.all(state != -1)), completeness_level=.95,
                       contrast_limit=completeness_limits(dmag_arr, detection))
        if checkpoint is not None:
            _save_sensitivity_checkpoint(checkpoint, payload)
        return payload

    result = snapshot()
    pending = np.argwhere(state == -1)
    t0 = time.perf_counter()
    print('Sensitivity fast: {} matched-PSF trials.'.format(len(pending)), flush=True)
    try:
        for number, (i, j, k) in tqdm(enumerate(pending, 1), total=len(pending)):
            started = time.perf_counter()
            theta = angles[i, j, k]
            x = target_xy[0]+dist_arr[i]/args.pxscale*np.cos(theta)
            y = target_xy[1]+dist_arr[i]/args.pxscale*np.sin(theta)
            if np.any(np.hypot(baseline_xy[:, 0]-x, baseline_xy[:, 1]-y) <= baseline_radius):
                state[i, j, k] = 2
                reasons[i, j, k] = 'preexisting_source'
            else:
                outcome = matched_injection_score(residual, popt, x, y, dmag_arr[j], core, settings['match_radius'])
                state[i, j, k] = int(outcome['recovered'])
                scores[i, j, k] = outcome['score']
                reasons[i, j, k] = outcome['reason']
                coverage[i, j, k] = outcome['valid_energy_fraction']
                baseline_scores[i, j, k] = outcome['baseline_score']
                increment_scores[i, j, k] = outcome['increment_score']
                matched_distances[i, j, k] = outcome['matched_distance']
            seconds[i, j, k] = time.perf_counter()-started
            # Save every 100 trials and on a controlled interruption. Rewriting
            # the entire expanded diagnostics cube per trial is unnecessary I/O.
            if number % 100 == 0 or number == len(pending):
                result = snapshot()
    except (KeyboardInterrupt, Exception):
        snapshot()
        raise
    return result


def sensitivity_cache_state(path, sources, target, popt, fake, args):
    """Validate parameters/data before reusing a curve; distinguish partial runs."""
    try:
        data = fits.getdata(os.path.join(args.root, '11_REDUCED', args.night, args.image))
        target_xy = (float(sources['xcentroid'][target]), float(sources['ycentroid'][target]))
        key = sensitivity_setup(data, popt, fake, target_xy, args)[-1]
        with np.load(path, allow_pickle=False) as saved:
            if 'cache_key' not in saved or str(saved['cache_key'].item()) != key:
                return 'incompatible'
            return 'complete' if bool(saved['complete'].item()) else 'resume'
    except (OSError, ValueError, KeyError):
        return 'incompatible'


def aperture_phot(data,positions,apsize=5, r_in=8, r_out=11, args=None):

	print(positions)
	
	# Background estimation in annulus:
	sigclip = SigmaClip(sigma=3.0, maxiters=10)
	annulus_aperture = CircularAnnulus(positions, r_in=r_in, r_out=r_out)
	bkg_stats = ApertureStats(data, annulus_aperture, sigma_clip=sigclip)
	bkg_mean = bkg_stats.median
	
	# Aperture photometry on target:
	aperture = CircularAperture(positions, r=apsize)
	phot_table = aperture_photometry(data, aperture, error=np.sqrt(data))
	
	# Background correction:
	aperture_area = aperture.area_overlap(data)
	total_bkg = bkg_stats.median * aperture.area
	phot_bkgsub = phot_table['aperture_sum'] - total_bkg
	phot_bkgsub_err = phot_table['aperture_sum_err']
	
	# Add this to the table photometry:
	phot_table['total_bkg'] = total_bkg
	phot_table['aperture_sum_bkgsub'] = phot_bkgsub
	phot_table['aperture_sum_bkgsub_err'] = phot_bkgsub_err

	# Add (uncalibrated) magnitudes:
	Zeropoint = 22
	phot_table['mag'] = Zeropoint + -2.5*log10(phot_bkgsub)
	phot_table['mag_err'] =  np.sqrt( (-2.5/(phot_bkgsub*np.log(10)) * phot_bkgsub_err )**2 )

	if 1:
		filename = get_filename(args)
		root = args.root
		night = args.night

		fig = plt.figure(figsize=(6.93,6.93))
		gs = gridspec.GridSpec(1,1, height_ratios=[1], width_ratios=[1])
		gs.update(left=0.12, right=0.97, bottom=0.08, top=0.97, wspace=0.12, hspace=0.08)

		norm = simple_norm(data, 'sqrt', percent=99)
		plt.imshow(data, norm=norm, interpolation='nearest')
		ap_patches = aperture.plot(color='white', lw=2,
								label='Photometry aperture')
		ann_patches = annulus_aperture.plot(color='red', lw=2,
											label='Background annulus')
		handles = (ap_patches[0], ann_patches[0])
		plt.legend(loc=(0.17, 0.05), facecolor='#458989', labelcolor='white',
				handles=handles, prop={'weight': 'bold', 'size': 11})	
		plt.xlabel('X (pixels)')
		plt.ylabel('Y (pixels)')
		plt.gca().invert_yaxis()
		plt.savefig(root+'/22_ANALYSIS/'+night+'/Summary_plots/'+filename+'__AperturePhot.pdf')
		plt.close()
	return phot_table

def findpeaks(data,npeaks=1, threshold=3):
	"""
	Find peaks in the image to detect the brightest star (considered as the target star)
	"""
	mean, median, std = sigma_clipped_stats(data, sigma=3.0)
	threshold = 1. * std
	tbl = find_peaks(data, threshold, npeaks=npeaks)

	return tbl

def radial_profile(data, center):
	"""
	Obtain the radial profile of the star
	center	: (x,y) location of the target star to get the radial profile
	"""
	y, x = np.indices((data.shape))
	r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
	neg = np.where((x - center[0]) < 0.0)[0]
	r = r.astype(int)

	tbin = np.bincount(r.ravel(), data.ravel())
	nr = np.bincount(r.ravel())
	radialprofile = tbin / nr

	return radialprofile

def get_filename(args):
	root = args.root
	night = args.night

	# Get information from image name and header
	file = os.path.splitext(args.image)[0]
	objname = file.split('_')[2]
	rate = file.split('_')[1]
	if len(root.split('_')) == 7: # For cases whith more than one obs per night (e.g., TOI-XXXX_1)
		idobs = file.split('_')[4]
		filter = file.split('_')[3]
	else:
		idobs = ''
		filter = file.split('_')[3]

	filename = file[14:]+'_'+rate

	return filename

def gaia_dr3_source_id(target: str) -> int | None:
	"""Returns the source_id of Gaia DR3, or None if it doesn't appear in SIMBAD."""
	ids = Simbad.query_objectids(target)
	if ids is None or len(ids) == 0:
		return None

	# Compatible with column names from different versions.
	column = next(c for c in ids.colnames if c.lower() == "id")

	for identifier in ids[column]:
		if isinstance(identifier, bytes):
			identifier = identifier.decode()
		match = re.fullmatch(r"Gaia DR3\s+(\d+)", str(identifier).strip())
		if match:
			return int(match.group(1))

	return None

def check_gaia(args, TOIname=None, target_name=None):
    """Optional Gaia crossmatch with a hard deadline for the entire lookup.

    A subprocess is killed and reaped on timeout, so no stalled query continues
    in the background. Failures return no associations, never stop photometry.
    """
    import json
    import subprocess
    import sys
    from pathlib import Path
    empty = (np.empty(0), np.empty(0), np.empty(0, dtype=np.int64), 0)
    timeout = float(getattr(args, 'GAIA_TIMEOUT', 30.))
    if not np.isfinite(timeout) or timeout <= 0:
        raise ValueError('GAIA_TIMEOUT must be finite and positive')
    command = [sys.executable, str(Path(__file__).with_name('astrasens_gaia.py'))]
    source_id = getattr(args, 'GDR3', None)
    tic = getattr(args, 'TIC', None)
    name = target_name or TOIname
    if source_id is not None:
        command.extend(['--source-id', str(source_id)])
        label = 'Gaia DR3 '+str(source_id)
    elif tic is not None:
        command.extend(['--tic', str(tic)])
        label = 'TIC '+str(tic)
    elif name:
        command.extend(['--target', str(name)])
        label = str(name)
    else:
        print('Gaia crossmatch skipped: no target name or identifier.', flush=True)
        return empty
    print(f'Gaia crossmatch: {label} (maximum wait {timeout:g}s)', flush=True)
    try:
        completed = subprocess.run(command, capture_output=True, text=True,
                                   timeout=timeout, check=False)
        if completed.returncode:
            raise RuntimeError(completed.stderr.strip()[-1200:] or 'lookup failed')
        result = json.loads(completed.stdout)
        return (np.asarray(result['delta_ra'], dtype=float),
                np.asarray(result['delta_dec'], dtype=float),
                np.asarray(result['gid'], dtype=np.int64), int(result['ngaia']))
    except subprocess.TimeoutExpired as exc:
        detail = exc.stderr or ''
        if isinstance(detail, bytes):
            detail = detail.decode(errors='replace')
        print(f'Gaia crossmatch timed out after {timeout:g}s; continuing without Gaia associations. '
              f'Last service messages: {detail[-600:]}', flush=True)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        print(f'Gaia crossmatch unavailable: {exc}; continuing without Gaia associations.', flush=True)
    return empty


def centroid_error(image, xp, yp):

	eimage = np.sqrt(image)
	Niter = 100
	xn, yn = np.zeros(Niter), np.zeros(Niter)
	for i in range(Niter):
		new_image = np.random.normal(image,eimage)
		xn[i], yn[i] = centroid_sources(new_image, xp, yp, box_size=11)
	
	expos, eypos = sigmaG(xn),sigmaG(yn)
	return expos, eypos


# ===========================================================================================================
# 						MAIN FUNCTIONS
# ===========================================================================================================

def sources(args):

	root = args.root
	night = args.night
	pxscale = args.pxscale  # arcsec/pixel

	# Read the AstraLux image
	hdu = fits.open(root+'/11_REDUCED/'+night+'/'+args.image)

	# Get information from image name and header
	file = os.path.splitext(args.image)[0]
	objname = file.split('_')[2]
	rate = file.split('_')[1]
	if len(root.split('_')) == 7: # For cases whith more than one obs per night (e.g., TOI-XXXX_1)
		idobs = file.split('_')[3]
		filter = file.split('_')[4]
	else:
		idobs = ''
		filter = file.split('_')[3]

	filename = file[14:]+'_'+rate

	data = hdu[0].data
	nx, ny = np.shape(data)

	# ==================================
	# Identify main peak
	# ==================================
	peaks   = findpeaks(data)
	center = [peaks['x_peak'],peaks['y_peak']]#[sources['xcentroid'][target], sources['ycentroid'][target]]#
	norm = ImageNormalize(stretch=SqrtStretch())
	#plt.imshow(data, origin='lower', norm=norm)
	#plt.scatter(peaks['x_peak'],peaks['y_peak'], lw=1.5, alpha=0.5,edgecolors='red',s=100,facecolors='none')
	#plt.show()

	# ==================================
	# PSF of the target
	# ==================================
	"""
	Fitting the PSF of the target with a mixed Gaussian + Lorentzian profile
	"""

	# ===== Get the radial profile and fit
	radprof = radial_profile(data, center)
	xradprof = np.arange(len(radprof))
	cumradprof = np.cumsum(radprof)
	cumradprof /= np.max(cumradprof)
	radprof_fwhm = np.interp(0.5,cumradprof,xradprof) / (2.*np.sqrt(2.*np.log(2.)))

	# ==== Get initial values:
	poptL, pcovL = curve_fit(psf_lorenz, xradprof[0:10], radprof[0:10], maxfev=10000)
	poptG, pcovG = curve_fit(psf_gauss, xradprof[20:], radprof[20:], maxfev=10000,
								p0=(100., 30., 0.0))
	# print(poptL)
	# print(poptG)
	# print(poptG[0]/poptL[0])

	if 0:
		plt.plot(xradprof,radprof,c='k',lw=2)
		plt.plot(xradprof,psf_gauss(xradprof,*poptG),c='green',ls=':',label='Gaussian')
		plt.plot(xradprof,psf_lorenz(xradprof,*poptL),c='red',ls='--',label='Lorentzian')
		plt.plot(xradprof[20:],radprof[20:],c='green',lw=2)
		plt.show()


	g0, g1, g2, g3 = poptL[0], poptL[1], poptG[1], 0.0
	l0, l2 = poptG[0]/poptL[0], poptL[2]

	if 0:
		plt.plot(xradprof,radprof,c='k',lw=2)
		popt=(2.*np.max(radprof),  0.0,    3.*radprof_fwhm, np.median(data), 0.10, 34.e-3/0.02723 )
		popt = (g0, g1, g2, g3, l0, l2)
		plt.plot(xradprof, psf_func(xradprof, *popt))

		plt.plot(xradprof,psf_gauss(xradprof,*poptG),c='green',ls=':',label='Gaussian')
		plt.plot(xradprof,psf_lorenz(xradprof,*poptL),c='red',ls='--',label='Lorentzian')
		plt.show()

		plt.plot(xradprof,cumradprof)
		plt.axvline(radprof_fwhm)
		plt.show()
		sys.exit()

	try:
		popt, pcov = curve_fit(psf_func, xradprof, radprof, maxfev=10000,
							p0=(g0, g1, g2, g3, l0, l2))
							# bounds = ([0.0   ,-nx,    1.0,-np.inf, 0.0,    0.0, 0.5],
							# 		  [np.inf, nx, np.inf, np.inf, 1.0, np.inf, 3.]), sigma=1./radprof )
	except:
		print(colored("\t --> Impossible to fit a Lorentzian+Gaussian profile. Trying only a Lorentzian...","yellow"))
		popt, pcov = curve_fit(psf_func_alternative, xradprof, radprof, maxfev=10000,
							p0=(0.0, np.max(radprof), 1.  ),
							bounds = ([-nx,    0.0,    0.01],
									  [nx, np.inf, 10.]) )


	if args.VERBOSE:
		fig = plt.figure()
		gs = gridspec.GridSpec(2,1, height_ratios=[1.,0.5], width_ratios=[1])
		gs.update(left=0.1, right=0.95, bottom=0.08, top=0.93, wspace=0.12, hspace=0.08)

		ax1 = plt.subplot(gs[0,0])
		plt.plot(xradprof,radprof,c='k',lw=2)
		plt.plot(xradprof, psf_func(xradprof, *popt))
		G = popt[0]*popt[-2]*np.exp(-(xradprof-popt[1])**2/(2*popt[2]**2)) + popt[3]
		L = popt[0] * 1./np.pi * 0.5*popt[-1]/((xradprof-popt[1])**2 + (0.5*popt[-1])**2)
		plt.plot(xradprof,G,c='green',ls=':',label='Gaussian')
		plt.plot(xradprof,L,c='red',ls='--',label='Lorentzian')
		plt.xscale('log')
		plt.legend()

		ax2 = plt.subplot(gs[1,0])
		plt.plot(xradprof,(radprof-(G+L))/np.max(radprof),c='k',lw=2)
		plt.xscale('log')
		plt.show()
		plt.close()

	# ===== Create the target fake PSF image
	x, y = np.meshgrid(np.linspace(0,nx,nx), np.linspace(0,ny,ny))
	xart,yart = center[0] , center[1] # sources['xcentroid'][target], sources['ycentroid'][target] #

	d = np.sqrt((x-xart)**2+(y-yart)**2)
	G = popt[0] *popt[-2] * np.exp(-( (d-popt[1])**2 / ( 2.0 * popt[2]**2 ) ) ) +popt[3]
	L = popt[0] * 1./np.pi * 0.5*popt[-1]/((d-popt[1])**2 + (0.5*popt[-1])**2)
	fake = G+L #* 10**(-2./2.5)
	# Measure the sampled core in pixels; do not confuse a centroid, Gaussian
	# sigma, or halo width with FWHM. Companion widths are fitted independently.
	yp, xp = int(peaks['y_peak'][0]), int(peaks['x_peak'][0])
	fwhm_target = target_core_fwhm(data, xp, yp)
	if args.VERBOSE:
		print('Measured target core FWHM: {:.3f} pixels'.format(fwhm_target))

	# ==================================
	# Identify target
	# ==================================
	myfwhm = 1.
	sources = np.array([])
	while len(sources) == 0:
		sources = find_sources(data,fwhm=myfwhm) #, signif=3. ,roundness=0.8,fwhm=1.5*fwhm_target
		myfwhm += 2

	target = np.argmax(sources['flux'])

	for col in sources.colnames:
		sources[col].info.format = '%.8g'  # for consistent table output

	if args.VERBOSE:
		print(sources)
		#positions = (sources['xcentroid'], sources['ycentroid'])
		#apertures = CircularAperture(positions, r=4.)
		norm = ImageNormalize(stretch=SqrtStretch())
		plt.imshow(data, origin='lower', norm=norm)
		#apertures.plot(color='blue', lw=1.5, alpha=0.5)
		plt.scatter(sources['xcentroid'], sources['ycentroid'], lw=1.5, alpha=0.5,edgecolors='red',s=100,facecolors='none')
		plt.scatter(sources['xcentroid'][target], sources['ycentroid'][target], lw=1.5, alpha=0.5,edgecolors='green',s=100,facecolors='none')		
		plt.show()
		plt.close()

	print("\t --> Main target identified...")


	# ==================================
	# Detect Source Companions
	# ==================================
	print("\t --> Looking for additional companions...\n")
	primary_xy = (float(sources['xcentroid'][target]), float(sources['ycentroid'][target]))
	residuals = companion_residuals(data, fake, primary_xy)
	print('Companion morphology: 2D Moffat + local plane; legacy -PA ignored.')
	psf_diagnostics = []
	sources2 = detect_companions(residuals, fwhm_target, args,
		diagnostics=psf_diagnostics, verbose=args.VERBOSE)
	if psf_diagnostics:
		ascii.write(Table(rows=psf_diagnostics),
			root+'/22_ANALYSIS/'+night+'/DetectedSources/'+filename+'_PSFDiagnostics.csv',
			format='csv', overwrite=True)
	print(len(np.shape(sources2)))

	if len(np.shape(sources2)) > 0 :

		target2 = np.argmax(sources2['flux'])
		# if 1:
		# 	plt.imshow(residuals, origin='lower', norm=norm)
		# 	plt.scatter(tbl['x_peak'],tbl['y_peak'],marker='x')
		# 	plt.scatter(sources2['xcentroid'], sources2['ycentroid'], lw=1.5, alpha=0.5,edgecolors='red',s=100,facecolors='none')
		# 	plt.scatter(sources2['xcentroid'][target], sources2['ycentroid'][target], lw=1.5, alpha=0.5,edgecolors='green',s=100,facecolors='none')
		# 	for s in sources2:
		# 		plt.text(s['xcentroid'], s['ycentroid'],s['id'])
		# 	plt.show()

		dist2 =  np.sqrt((sources2['xcentroid']-sources['xcentroid'][target])**2+
						 (sources2['ycentroid']-sources['ycentroid'][target])**2)

		# Aperture photometry
		positions = [(sources['xcentroid'][target],sources['ycentroid'][target])]
		for s,source in enumerate(sources2): positions.append((source['xcentroid'],source['ycentroid']))
		phot_table = aperture_phot(data,positions,args=args)

		print(colored("\t --> "+str(len(sources2))+" companion(s) found...","yellow"))
		print(phot_table)

		# Gaia sources within 5 arcsec
		if "TOI" in objname: 
			TOIname = objname
		else:
			TOIname = None
		delta_ra, delta_dec, gid, ngaia = check_gaia(args,TOIname=TOIname, target_name=objname)

		

		id, sep, esep = [], [], [] #np.zeros(len(sources)),np.zeros(len(sources)),np.zeros(len(sources)),np.zeros(len(sources))
		PA, ePA       = [], []
		xpos, ypos    = [], []
		expos, eypos  = [], []
		dmag, edmag   = [], []
		gaiacount, gaiasep = [], []
		sid = 1
		for s,source in enumerate(sources2):
			if dist2[s] > 4.:
				id.append(sid)
				_xpos, _ypos = source["xcentroid"], source["ycentroid"]
				xpos.append(_xpos)
				ypos.append(_ypos)

				# Get uncertainty on position:
				_expos, _eypos = centroid_error(residuals,_xpos, _ypos)
				expos.append(_expos)
				eypos.append(_eypos)

				# Separation
				Dx = source["xcentroid"]-sources['xcentroid'][target]
				Dy = source["ycentroid"]-sources['ycentroid'][target]
				separation = args.pxscale * np.sqrt(Dx**2+Dy**2)
				sep.append(separation)
				eseparation = 2* 2*args.pxscale**2/separation * np.sqrt((Dx*_expos)**2+(Dy*_eypos)**2)
				esep.append(eseparation)

				# Position angle (PA)
				_PA = np.arctan(Dy/Dx) * 180./np.pi
				if source["xcentroid"] < center[0]: _PA += 180.
				PA.append(_PA)
				_ePA = 2* np.sqrt( (_expos/(Dx+Dy))**2 + (Dx*_eypos/(Dy**2+Dx*Dy))**2 )* 180./np.pi
				ePA.append(_ePA)

				# Contrast
				dmag.append(phot_table["mag"][s+1]-phot_table["mag"][0])
				edmag.append(np.sqrt(phot_table["mag_err"][s+1]**2+phot_table["mag_err"][0]**2))
				# Check Gaia counterpart:
				if ngaia > 1:
					delta_x_comp = (source['xcentroid']-sources['xcentroid'][target])*pxscale
					delta_y_comp = (source['ycentroid']-sources['ycentroid'][target])*pxscale
					sep2gaia = np.sqrt((delta_x_comp-delta_ra)**2 + (delta_y_comp-delta_dec)**2)
					match_gaia = np.where(sep2gaia < 0.3)[0] # < 0.3 arcsec
					if len(match_gaia) == 0:
						gaiacount.append(-99)
						gaiasep.append(-99)
					else:
						gaiacount.append(gid[match_gaia][0])
						gaiasep.append(sep2gaia[match_gaia][0])
				else:
					gaiacount.append(-99)
					gaiasep.append(-99)

				sid += 1

		table = Table([id, sep, esep, PA, ePA, dmag, edmag, xpos, expos, ypos, eypos, gaiacount,gaiasep], names=['#id', 'sep', 'esep', 'PA', 'ePA', 'dmag', 'dmag_err','xpix', 'expix','ypix', 'eypix','GaiaDR3_counterpart','Gaiasep_arcsec'])
		format_output, suffix = 'csv', '.csv'
		if args.IPAC: format_output, suffix = 'ipac', '.dat'
		ascii.write(table, root+'/22_ANALYSIS/'+night+'/DetectedSources/'+filename+'_Sources'+suffix,format=format_output,overwrite=True)

	else:

		print("\t --> No additional companions found")


	if args.VERBOSE:
		norm = ImageNormalize(stretch=SqrtStretch())
		plt.imshow(residuals, origin='lower', norm=norm )
		if len(np.shape(sources2)) > 0:
			plt.scatter(sources2['xcentroid'], sources2['ycentroid'], lw=1.5, alpha=0.5,edgecolors='red',s=200,facecolors='none')
		plt.scatter(sources['xcentroid'][target], sources['ycentroid'][target], lw=1.5,edgecolors='green',s=100,facecolors='none')
		plt.show()
		plt.close()


	np.savez(root+'/22_ANALYSIS/'+night+'/DetectedSources/'+filename+'__Sources',
														sources=sources, target=target, popt=popt,
														fake=fake, myfwhm=myfwhm,center=center,
														sources2=sources2)

	return sources,target, popt, fake, myfwhm, center, sources2

def sensitivity(sources, target, popt, fake, myfwhm, args):
    """Persist/resume completeness measured with the shared companion detector.

    myfwhm remains in the signature for callers; the old primary-DAO search
    width is intentionally not used to recover companions.
    """
    data = fits.getdata(os.path.join(args.root, '11_REDUCED', args.night, args.image))
    primary_xy = (float(sources['xcentroid'][target]), float(sources['ycentroid'][target]))
    output = os.path.join(args.root, '22_ANALYSIS', args.night, 'Sensitivity',
                          get_filename(args)+'__Sensitivity.npz')
    method = getattr(args, 'SENS_METHOD', 'matched')
    runner = compute_sensitivity if method == 'blind' else compute_sensitivity_fast
    return runner(data, popt, fake, primary_xy, args, checkpoint=output,
                  restart=bool(getattr(args, 'FORCE', False)))
