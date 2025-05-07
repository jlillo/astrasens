import os
from sys import platform
if platform == 'linux2': plt.switch_backend('agg')
import argparse

import numpy as np
from numpy import unravel_index
from pylab import *
import time
import progressbar
import tqdm
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
Simbad.add_votable_fields('pmra', 'pmdec')
from astroquery.gaia import Gaia
Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"  # Reselect Data Release 3, default

from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings("ignore") 

import astrasens_plot  as plotting
import jlillo_pypref
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


def find_sources(data, XYcoords=None, fwhm=10., min_sharpness=0.8, roundness = 0.1, signif=5.0, fluxmin = 1., SENS=False, VERBOSE=False, COMPANIONS=False):
	"""
	Function to detect stars above 5-sigma of the sky in the image
	- Sources in the edge are removed
	- Sources not round are removed
	- Sources with total_flux < 5*sky_std are removed
	"""
	mean, median, std = sigma_clipped_stats(data, sigma=3.0)

	if COMPANIONS: 
		tmp=0.2
		# print(XYcoords)
		# XYcoords = ((669,713),) + XYcoords
		# min_sharpness = 0.80
		# fluxmin = 18
		# signif=0.1
		# fwhm=20
		# roundness=0.25
	
	if XYcoords != None:
		daofind = DAOStarFinder(xycoords=XYcoords,fwhm=fwhm, threshold=signif*np.abs(std), )
		if VERBOSE:
			plt.imshow(data)
			for x in XYcoords:
				plt.scatter(x[0],x[1],marker='x',c='k')	
	else:
		daofind = DAOStarFinder(fwhm=float(fwhm), threshold=signif*np.abs(std))
	_sources = daofind.find_stars(data-median)

	nx, ny = np.shape(data)

	if VERBOSE:
		print(_sources)
		plt.scatter(_sources['xcentroid'],_sources['ycentroid'],marker='x',c='red')
		for ss in _sources: plt.text(ss['xcentroid'],ss['ycentroid'],ss['id'],c='red')

	# Remove close companions an no-star like sources

	if ((COMPANIONS == True) & (len(np.shape(_sources)) > 0) ):
		if len(_sources) > 0:
			min_separation = 4 # pixels
			keep = np.full(len(_sources),True)
			
			# Separation betwee sources
			for ss,s in enumerate(_sources):
				if keep[ss] == True:
					sep = np.sqrt((s['xcentroid'] -_sources['xcentroid'])**2+(s['ycentroid']-_sources['ycentroid'])**2)
					matches = np.where(sep < min_separation)[0]
					if len(matches) > 1:
						keep[matches] = False
						# Keep the brightest:
						keepthis = np.argmax(_sources['peak'][matches])
						keep[matches[keepthis]] = True

			print(_sources[keep])
			# Star-like source
			for ss,s in enumerate(_sources):
				if keep[ss] == True:
					xc,yc = int(s['xcentroid']),int(s['ycentroid'])
					xradprof = np.arange(40)
					horprof = np.sum(data[yc-3:yc+3,xc-20:xc+20],axis=0)
					verprof = np.sum(data[yc-20:yc+20,xc-3:xc+3],axis=1)
					bounds = ([0,   0 ,    0., -np.inf,-np.inf], [10000, 40, np.inf, np.inf,np.inf] )
					# Vertical profile
					try:
						p0 = (np.max(verprof),20, 0. , np.min(verprof), 0.0 )
						popt, pcov = curve_fit(psf_lorenz2, xradprof, verprof, maxfev=10000, p0=p0, bounds = bounds )
						Vwidth = popt[2]
						Vcenter = popt[1]
					except:
						Vwidth = 100
						Vcenter = 0
					# Horizontal profile
					try:
						p0 = (np.max(horprof),20, 0. , np.min(horprof), 0.0 )
						popt, pcov = curve_fit(psf_lorenz2, xradprof, horprof, maxfev=10000, p0=p0, bounds = bounds )
						Hwidth  = popt[2] 
						Hcenter = popt[1]
					except:
						Hwidth = 100
						Hcenter = 0

					if ((Hwidth > 20) | (Vwidth > 20)):
						keep[ss] = False	
					if ((np.abs(Hcenter-20) > 3) | (np.abs(Vcenter-20) > 3)) :
						keep[ss] = False

					# print(s['id'],Hwidth,Hcenter,Vwidth,Vcenter)
					# if s['id'] == 13:
					# 	plt.figure(2)
					# 	plt.plot(xradprof,horprof)
					# 	plt.plot(xradprof,verprof)
					# 	plt.axvline(Hcenter,ls=':')
					# 	plt.axvline(Vcenter,ls=':')
					# 	plt.show()

			_sources = _sources[keep]


	if VERBOSE:
		plt.scatter(_sources['xcentroid'],_sources['ycentroid'],marker='x',c='white')

	# if SENS:
	# 	plt.imshow(data)
	# 	plt.show()
	# 	sys.exit()

	if len(np.shape(_sources)) > 0: #len(np.atleast_1d(_sources)) > 0:
		# Remove border sources:
		if SENS:
			noborder = np.where((_sources['xcentroid'] > 0.05*nx) & (_sources['xcentroid'] < 0.95*nx) &
								(_sources['ycentroid'] > 0.05*ny) & (_sources['ycentroid'] < 0.95*ny))[0]
		elif COMPANIONS == True:
			print(roundness,min_sharpness)
			noborder = np.where((_sources['xcentroid'] > 0.05*nx) & (_sources['xcentroid'] < 0.95*nx) &
								(_sources['ycentroid'] > 0.05*ny) & (_sources['ycentroid'] < 0.95*ny) &
								(_sources['peak'] > fluxmin) &
								((np.abs(_sources['roundness1']) < roundness) | (np.abs(_sources['roundness2']) < roundness)) &
								(_sources['sharpness'] > min_sharpness))[0] # & (_sources['peak'] > 10.)	)[0]
		else:
			noborder = np.where((_sources['xcentroid'] > 0.05*nx) & (_sources['xcentroid'] < 0.95*nx) &
								(_sources['ycentroid'] > 0.05*ny) & (_sources['ycentroid'] < 0.95*ny) &
								(_sources['peak'] > fluxmin))[0] #&
								# (np.abs(_sources['roundness2']) < roundness) &
								# (_sources['sharpness'] > min_sharpness))[0] # & (_sources['peak'] > 10.)	)[0]

		from prettytable import PrettyTable
		table = PrettyTable()
		table.field_names = ["Xcentroid", "Ycentroid", "Fpeak", "Roundness1","Roundness2","Sharpness"]
		for ss in _sources:
			table.add_row([ss['xcentroid'],ss['ycentroid'],ss['peak'],ss['roundness1'],ss['roundness2'],ss['sharpness']])

		if len(noborder) == 0: 
			sources = None
			print(colored("ERROR: there are no sources meeting the requested criteria:","red"))
			print(colored("\t x in ["+str(0.05*nx)+","+str(0.95*nx)+"]" ,"red"))
			print(colored("\t y in ["+str(0.05*ny)+","+str(0.95*ny)+"]" ,"red"))
			print(colored("\t roundness < "+str(roundness),"red"))
			print(colored("\t sharpness > "+str(min_sharpness),"red"))
			print(table)
			print(colored("\t --> Try modifying the min_sharpness and roundness parameters","blue"))
			# sys.exit()			
		else:
			sources = _sources[noborder]

		if VERBOSE:
			print(table)	

	else:
		sources = _sources

	if VERBOSE:
		print(_sources)
		print(sources)
		if sources is not None:
			plt.scatter(sources['xcentroid'],sources['ycentroid'],marker='x',c='gold')
			plt.show()
			plt.close()

	return sources

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

def check_gaia(args,TOIname=None):
	'''
		Returns the Gaia sources within 5 arcsec around the target

	'''

	print("\t --> Querying Gaia to look for the detected companions...")

	if args.GDR3 is not None:
		gaia_id = args.GDR3
		result = plotting.get_gaia_data_from_simbad(args.GDR3)
		ra,dec = result['ra'].value.data[0], result['dec'].value.data[0]
	elif ((TOIname == None) & (args.TIC is not None)):
		tic = args.TIC
	elif TOIname is not None:
		# Read AstraLux targets
		ast = np.genfromtxt('/Users/lillo_box/00_projects/11__HighResTESS/targets_table_astralux_TIC.csv',delimiter=',',encoding='utf-8',dtype=None,names=True)
		try:
			toi = np.abs(int(float(TOIname[3:])))
		except:
			toi = int(float(TOIname[4:]))
		this = np.where(ast['TOI'] == toi)[0]
		this = np.atleast_1d(this)
		tic = str(ast['TIC'][this[0]])

		# Get TIC corrdinates and Gaia DR3 ID
		ra,dec = plotting.get_coord(tic)
		gaia_id, mag = plotting.get_dr2_id_from_tic(tic)
		gaia_id = plotting.dr3_from_dr2(gaia_id)

	else: 
		print(colored('\t --> **ERROR** You must specify either a TIC name (--TIC), a TOI name, or a Gaia DR3 ID (--GDR3)','red'))
		print('Exiting...')
		sys.exit()


	# Search for Gaia sources
	coord = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame='fk5')
	Gaia.ROW_LIMIT = -1
	j = Gaia.cone_search_async(coord, radius=u.Quantity(5.0, u.arcsec))
	gaiares = j.get_results()
	gaiares.pprint()

	# Position on CCD
	ngaia = len(gaiares['SOURCE_ID'].value.data) 
	if ngaia > 1:
		gaia_companions = {}
		targ = np.where(gaiares['SOURCE_ID'].astype(int) == int(gaia_id))[0]
		delta_ra,delta_dec,gid = [],[],[]
		for i,row in enumerate(gaiares):
			if row['SOURCE_ID'] == gaia_id: continue
			delta_ra.append(-1*(row['ra'] - gaiares['ra'][targ].data)[0] * 3600)
			delta_dec.append((row['dec'] - gaiares['dec'][targ].data )[0] * 3600)
			gid.append(row['SOURCE_ID'])
		delta_ra,delta_dec,gid = np.array(delta_ra), np.array(delta_dec), np.array(gid)
	else:
		delta_ra,delta_dec,gid = 0,0,0

	return delta_ra,delta_dec,gid,ngaia

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
	pxscale = 0.02327 # arcsec/pixel

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
	residuals = data-fake
	square = 5
	residuals[int(sources['ycentroid'][target]-square):int(sources['ycentroid'][target]+square),
			  int(sources['xcentroid'][target]-square):int(sources['xcentroid'][target]+square)	] = np.nan


	# Find peaks in residuals image
	tbl = findpeaks(residuals,npeaks=1000)
	if args.VERBOSE:
		print(tbl)

	XYcoords = []
	for i in tbl: XYcoords.append((i['x_peak'],i['y_peak']))

	# Get info on the detected peaks
	myfwhm2 = popt[2]
	if popt[2] > 15:
		myfwhm2 = 10
	elif popt[2] < 5:
		myfwhm2 = 10

	sources2 = find_sources(residuals,XYcoords=tuple(XYcoords), fwhm=myfwhm2, roundness=args.PARS[0], min_sharpness=args.PARS[1], \
							fluxmin=2., signif=2., COMPANIONS=True, VERBOSE=args.VERBOSE) # ,roundness=0.8

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
		delta_ra, delta_dec, gid, ngaia = check_gaia(args,TOIname=TOIname)

		

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
				separation = 0.02327 * np.sqrt(Dx**2+Dy**2)
				sep.append(separation)
				eseparation = 2* 2*pxscale**2/separation * np.sqrt((Dx*_expos)**2+(Dy*_eypos)**2)
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

def sensitivity(sources, target, popt, fake,myfwhm, args):

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
	if args.WINDOW is not None:
		pxscale = 0.02327
		size = int(args.WINDOW  / 2) 
		maxdist  = np.min([ (nx-sources['xcentroid'][target])*pxscale-0.2,
							(ny-sources['ycentroid'][target])*pxscale-0.2,
							 sources['xcentroid'][target]*pxscale-0.2,
							 sources['ycentroid'][target]*pxscale-0.2,
							 size])-2*pxscale #3. # arcsec
		print("\t Windowing activated for a window size of "+str(round(maxdist,1))+" arcsec")
		xt,yt =  sources['xcentroid'][target], sources['ycentroid'][target]

		data = data[int(yt-maxdist/pxscale):int(yt+maxdist/pxscale),int(xt-maxdist/pxscale):int(xt+maxdist/pxscale)]
		fake = fake[int(yt-maxdist/pxscale):int(yt+maxdist/pxscale),int(xt-maxdist/pxscale):int(xt+maxdist/pxscale)]
		ny, nx = np.shape(data)
		xt,yt =  maxdist/pxscale,maxdist/pxscale
		if args.VERBOSE:
			plt.imshow(data)
			plt.scatter(xt,yt,marker='x',c='k')
			plt.show()
			sys.exit()
	else:
		xt,yt =  sources['xcentroid'][target], sources['ycentroid'][target]
		maxdist  = np.min([ (nx-sources['xcentroid'][target])*0.02327-0.2,
							(ny-sources['ycentroid'][target])*0.02327-0.2,
							6. ])#3. # arcsec

	# ==================================
	# Get sensitivity curve
	# ==================================
	print('\t --> Calculating SENSITIVITY curve')

	Ndist, Dmag_max, Dmag_step = int(args.SENSPAR[0]), float(args.SENSPAR[1]), float(args.SENSPAR[2])
	Nstars = int(args.SENSPAR[3])

	dist_arr = np.logspace(np.log10(0.1), np.log10(maxdist),Ndist)

	maxmag   = Dmag_max # delta_mag maximum
	magstep  = Dmag_step # mag
	dmag_arr = np.linspace(maxmag, 0.0, int(maxmag/magstep))
	nstars   = Nstars 

	detection = np.zeros((len(dist_arr),len(dmag_arr)))


	for i,dd in enumerate(progressbar.progressbar(dist_arr)):
		for j,dm in enumerate(dmag_arr):
			# print(i,j)

			# --- Include the ARTIFICIALLY added star
			thetas = np.random.uniform(low=0.0,high=2.*np.pi,size=nstars)
			yes = 0.

			for theta in thetas:
				xart = xt + dd/0.02327*np.cos(theta)
				yart = yt + dd/0.02327*np.sin(theta)

				x, y = np.meshgrid(np.linspace(0,nx,nx), np.linspace(0,ny,ny))
				d = np.sqrt((x-xart)**2+(y-yart)**2)
				d = np.sqrt((x-xart)**2+(y-yart)**2)
				G = popt[0] *popt[-2] * np.exp(-( (d-popt[1])**2 / ( 2.0 * popt[2]**2 ) ) ) +popt[3]
				L = popt[0] * 1./np.pi * 0.5*popt[-1]/((d-popt[1])**2 + (0.5*popt[-1])**2)
				image_Art = (G+L) * 10**(-dm/2.5)

				# === Check if the star has been detected
				# XYcoords = []
				# XYcoords.append((xart,yart))
				detected = find_sources(image_Art+data-fake, fwhm=myfwhm, SENS=True)
				if len(np.shape(detected)) > 0:
					dist = np.sqrt((xart-detected['xcentroid'])**2+(yart-detected['ycentroid'])**2)

					# === Identify target among detected sources
					if args.VERBOSE:
						norm = ImageNormalize(stretch=SqrtStretch())
						plt.imshow(image_Art+data-fake, cmap='Greys', origin='lower', norm=norm)
						plt.scatter(detected['xcentroid'], detected['ycentroid'], marker='x',s=100,color='green')
						plt.scatter(xt, yt, marker='s',s=100, facecolors='none',edgecolors='k')
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

	print('Successfully finished...')
