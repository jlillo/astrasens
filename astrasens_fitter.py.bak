import numpy as np
from astropy.io import fits
from sys import platform
import matplotlib.pyplot as plt
if platform == 'linux2': plt.switch_backend('agg')
import matplotlib.gridspec as gridspec # GRIDSPEC !
from numpy import unravel_index
from matplotlib.colors import LogNorm
from astropy.io import ascii
from pylab import *
import matplotlib as mpl
import matplotlib.patches as patches
from matplotlib.colorbar import Colorbar 
import argparse
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils import datasets	
from photutils import DAOStarFinder
import matplotlib.pyplot as plt
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils import CircularAperture
import time
from photutils import find_peaks
from scipy.optimize import curve_fit
from astropy.io import ascii
import os
from astropy.table import Table, Column, MaskedColumn
from scipy.interpolate import interp1d
import jlillo_pypref

"""
	Automatic analysis of AstraLux images to get the sensitivity curve.
	
	===== SYNTAX ===== 
	date:	Night to be reduced: YYMMDD
	-SF	:	Setup file location can be modified as --SF path_to_file
	
	===== HISTORY ===== 
	2019/05/08		jlillobox		First version released
	
"""

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


def find_sources(data, fwhm=10., min_sharpness=0.5, roundness = 0.3, signif=5.0, fluxmin = 20., SENS=False, VERBOSE=False):
	"""
	Function to detect stars above 5-sigma of the sky in the image
	- Sources in the edge are removed
	- Sources not round are removed
	- Sources with total_flux < 5*sky_std are removed
	"""	
	mean, median, std = sigma_clipped_stats(data, sigma=3.0) 

	daofind = DAOStarFinder(fwhm=fwhm, threshold=signif*np.abs(std))
	_sources = daofind(data - median)
	
	nx, ny = np.shape(data)
	
	if len(_sources) > 0:
	# Remove border sources:
		noborder = np.where((_sources['xcentroid'] > 0.1*nx) & (_sources['xcentroid'] < 0.9*nx) &
							(_sources['ycentroid'] > 0.1*ny) & (_sources['ycentroid'] < 0.9*ny) &
							(np.abs(_sources['roundness2']) < roundness) & (_sources['flux'] > fluxmin) &
							(_sources['sharpness'] > min_sharpness) & (_sources['peak'] > 10.)	)[0]
		if SENS:
			noborder = np.where((_sources['xcentroid'] > 0.05*nx) & (_sources['xcentroid'] < 0.95*nx) &
								(_sources['ycentroid'] > 0.05*ny) & (_sources['ycentroid'] < 0.95*ny))[0]
			
	
		sources = _sources[noborder]
	else:
		sources = _sources

	return sources

def findpeaks(data):
	"""
	Find peaks in the image to detect the brightest star (considered as the target star)
	"""
	mean, median, std = sigma_clipped_stats(data, sigma=3.0)
	threshold = median + (5. * std)
	tbl = find_peaks(data, threshold, npeaks=1)
	return tbl

def radial_profile(data, center):
	"""
	Obtain the radial profile of the star
	center	: (x,y) location of the target star to get the radial profile
	"""
	y, x = np.indices((data.shape))
	r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
	neg = np.where((x - center[0]) < 0.0)[0]
	r = r.astype(np.int)
	
	tbin = np.bincount(r.ravel(), data.ravel())
	nr = np.bincount(r.ravel())
	radialprofile = tbin / nr
	
	return radialprofile 




def sources(args):

	root = args.root
	night = args.night		

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
	print poptL
	print poptG
	print poptG[0]/poptL[0]

# 	plt.plot(xradprof,radprof,c='k',lw=2)
# 	plt.plot(xradprof,psf_gauss(xradprof,*poptG),c='green',ls=':',label='Gaussian')
# 	plt.plot(xradprof,psf_lorenz(xradprof,*poptL),c='red',ls='--',label='Lorentzian')
# 	plt.plot(xradprof[20:],radprof[20:],c='green',lw=2)
# 	plt.show()


	g0, g1, g2, g3 = poptL[0], poptL[1], poptG[1], 0.0
	l0, l2 = poptG[0]/poptL[0], poptL[2]


# 	plt.plot(xradprof,radprof,c='k',lw=2)
# 	popt=(2.*np.max(radprof),  0.0,    3.*radprof_fwhm, np.median(data), 0.10, 34.e-3/0.02723 )
# 	popt = (g0, g1, g2, g3, l0, l2)
# 	plt.plot(xradprof, psf_func(xradprof, *popt))
# 	
# 	plt.plot(xradprof,psf_gauss(xradprof,*poptG),c='green',ls=':',label='Gaussian')
# 	plt.plot(xradprof,psf_lorenz(xradprof,*poptL),c='red',ls='--',label='Lorentzian')
# 	plt.show()


# 	plt.plot(xradprof,cumradprof)
# 	plt.axvline(radprof_fwhm)
# 	plt.show()
# 	sys.exit()
	try:
		popt, pcov = curve_fit(psf_func, xradprof, radprof, maxfev=10000,
							p0=(g0, g1, g2, g3, l0, l2))
#							bounds = ([0.0   ,-nx,    1.0,-np.inf, 0.0,    0.0, 0.5],
#									  [np.inf, nx, np.inf, np.inf, 1.0, np.inf, 3.]), sigma=1./radprof )
	except:
		print "Impossible to fit a Lorentzian+Gaussian profile. Trying only a Lorentzian..."
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
		plt.legend()
		
		ax2 = plt.subplot(gs[1,0]) 
		plt.plot(xradprof,(radprof-(G+L))/np.max(radprof),c='k',lw=2)
		plt.show()
		plt.close()
	
	# ===== Create the target fake PSF image
	x, y = np.meshgrid(np.linspace(0,nx,nx), np.linspace(0,ny,ny))
	xart,yart = center[0] , center[1] # sources['xcentroid'][target], sources['ycentroid'][target] #
	d = np.sqrt((x-xart)**2+(y-yart)**2)
	G = popt[0] *popt[-2] * np.exp(-( (d-popt[1])**2 / ( 2.0 * popt[2]**2 ) ) ) +popt[3]
	L = popt[0] * 1./np.pi * 0.5*popt[-1]/((d-popt[1])**2 + (0.5*popt[-1])**2)
	fake = G+L #* 10**(-2./2.5)
	if popt[2] < 15:
		fwhm_target = 2.*np.sqrt(2.*np.log(2.))* popt[2]
	else:
		fwhm_target = 2.*popt[5]
	fwhm_target = 2.*popt[5]

	# ==================================
	# Identify target 
	# ==================================
	myfwhm = 1.
	sources = np.array([])
	while len(sources) == 0:
		sources = find_sources(data,fwhm=myfwhm) #, signif=3. ,roundness=0.8,fwhm=1.5*fwhm_target
		myfwhm += 2
	
	print myfwhm
	target = np.argmax(sources['flux'])
	
	for col in sources.colnames:    
		sources[col].info.format = '%.8g'  # for consistent table output

	if args.VERBOSE:
		print sources    
		#positions = (sources['xcentroid'], sources['ycentroid'])
		#apertures = CircularAperture(positions, r=4.)
		norm = ImageNormalize(stretch=SqrtStretch())
		plt.imshow(data, origin='lower', norm=norm)
		#apertures.plot(color='blue', lw=1.5, alpha=0.5)
		plt.scatter(sources['xcentroid'], sources['ycentroid'], lw=1.5, alpha=0.5,edgecolors='red',s=100,facecolors='none')
		plt.scatter(sources['xcentroid'][target], sources['ycentroid'][target], lw=1.5, alpha=0.5,edgecolors='green',s=100,facecolors='none')
		plt.show()
		plt.close()	



	# ==================================
	# Detect Source Companions
	# ==================================
	residuals = data-fake
	residuals[int(sources['ycentroid'][target]-4):int(sources['ycentroid'][target]+4),
			  int(sources['xcentroid'][target]-4):int(sources['xcentroid'][target]+4)	] = np.nan

	sources2 = find_sources(residuals,fwhm=myfwhm/2., roundness=0.15, fluxmin=5., signif=3.) # ,roundness=0.8
	if len(sources2) > 0:
		dist2 =  np.sqrt((sources2['xcentroid']-sources['xcentroid'][target])**2+
						 (sources2['ycentroid']-sources['ycentroid'][target])**2)
	
		id, sep, PA, dmag = [], [], [], [] #np.zeros(len(sources)),np.zeros(len(sources)),np.zeros(len(sources)),np.zeros(len(sources))
		sid = 1
		for s,source in enumerate(sources2):
			if dist2[s] > 4.:
				id.append(sid)
				sep.append(0.02327 * np.sqrt((source["xcentroid"]-center[0])**2+(source["ycentroid"]-center[1])**2))
				_PA = np.arctan((source["ycentroid"]-center[1])/(source["xcentroid"]-center[0])) * 180./np.pi
				if source["xcentroid"] < center[0]: _PA += 180.
				PA.append(_PA)
				dmag.append(source["mag"] - sources["mag"][target])
				sid += 1
	
		table = Table([id, sep, PA, dmag], names=['id', 'sep', 'PA', 'dmag'])
		ascii.write(table, root+'/22_ANALYSIS/'+night+'/DetectedSources/'+filename+'_Sources.dat',format='ipac',overwrite=True)
	print sources2


	if args.VERBOSE:
		norm = ImageNormalize(stretch=SqrtStretch())
		plt.imshow(residuals, origin='lower', norm=norm )
		if len(sources2) > 0:
			plt.scatter(sources2['xcentroid'], sources2['ycentroid'], lw=1.5, alpha=0.5,edgecolors='red',s=200,facecolors='none')
		plt.scatter(sources['xcentroid'][target], sources['ycentroid'][target], lw=1.5,edgecolors='green',s=100,facecolors='none')
		plt.show()
		plt.close()

	
	np.savez(root+'/22_ANALYSIS/'+night+'/DetectedSources/'+filename+'__Sources',
														sources=sources, target=target, popt=popt, 
														fake=fake, myfwhm=myfwhm)
	
	return sources,target, popt, fake, myfwhm, center, sources2

def sensitivity(sources, target, popt, fake,myfwhm, center, sources2, args):

	root = args.root
	night = args.night		

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
	# Get sensitivity curve
	# ==================================

	maxdist  = np.min([ (nx-sources['xcentroid'][target])*0.02327-0.8, 
						(ny-sources['ycentroid'][target])*0.02327-0.8, 
						6. ])#3. # arcsec
	dist_arr = np.logspace(np.log10(0.1), np.log10(maxdist),20)
	print dist_arr
	
	maxmag   = 10. # delta_mag maximum
	magstep  = 0.5 # mag
	dmag_arr = np.linspace(maxmag, 0.0, maxmag/magstep)
	
	
	nstars = 10
	
	detection = np.zeros((len(dist_arr),len(dmag_arr)))
	
	for i,dd in enumerate(dist_arr):
		for j,dm in enumerate(dmag_arr):
			print i,j
		
			# --- Include the ARTIFICIALLY added star			
			thetas = np.random.uniform(low=0.0,high=2.*np.pi,size=nstars)
			yes = 0.
			
			for theta in thetas:
				xart = sources['xcentroid'][target] + dd/0.02327*np.cos(theta)
				yart = sources['ycentroid'][target] + dd/0.02327*np.sin(theta)
			
				x, y = np.meshgrid(np.linspace(0,nx,nx), np.linspace(0,ny,ny))
				d = np.sqrt((x-xart)**2+(y-yart)**2)
				d = np.sqrt((x-xart)**2+(y-yart)**2)
				G = popt[0] *popt[-2] * np.exp(-( (d-popt[1])**2 / ( 2.0 * popt[2]**2 ) ) ) +popt[3]
				L = popt[0] * 1./np.pi * 0.5*popt[-1]/((d-popt[1])**2 + (0.5*popt[-1])**2)
				image_Art = (G+L) * 10**(-dm/2.5)

				# === Check if the star has been detected
				detected = find_sources(image_Art+data-fake, fwhm=myfwhm, SENS=True)
				if len(detected) > 0:
					dist = np.sqrt((xart-detected['xcentroid'])**2+(yart-detected['ycentroid'])**2)
				
					# === Identify target among detected sources
					if args.VERBOSE:
						norm = ImageNormalize(stretch=SqrtStretch())
						plt.imshow(image_Art+data-fake, cmap='Greys', origin='lower', norm=norm)
						plt.scatter(detected['xcentroid'], detected['ycentroid'], marker='x',s=100,color='green')
						plt.scatter(sources['xcentroid'][target], sources['ycentroid'][target], marker='s',s=100, facecolors='none',edgecolors='k')
						plt.scatter(xart,yart,lw=1.5, facecolors='none',edgecolors='blue',s=100)
						plt.show()
						sys.exit()

					if len(np.where(dist < 1.0)[0]) != 0:
						yes += 1.

# 				plt.imshow(image_Art+data)
# 				plt.show()
# 				sys.exit()
			
			detection[i,j] = yes/nstars
							

	np.savez(root+'/22_ANALYSIS/'+night+'/Sensitivity/'+filename+'__Sensitivity',detection=detection,dist_arr=dist_arr,dmag_arr=dmag_arr)

	print 'Successfully finished...'


			
	
	







