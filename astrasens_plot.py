import numpy as np
from sys import platform
if platform == 'linux2': plt.switch_backend('agg')
from numpy import unravel_index
from pylab import *
import argparse
import time
import os
import jlillo_pypref

from photutils import datasets
from photutils import DAOStarFinder
from photutils import CircularAperture
from photutils import find_peaks

import matplotlib as mpl
import matplotlib.patches as patches
from matplotlib.colorbar import Colorbar
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec # GRIDSPEC !
from matplotlib.colors import LogNorm

from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import SqrtStretch,LinearStretch
import astropy.visualization as stretching
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Table, Column, MaskedColumn
from astropy.io import fits
from astropy.io import ascii
from astropy.coordinates import SkyCoord
import astropy.units as u

from astroquery.mast import Catalogs
from astroquery.simbad import Simbad
Simbad.add_votable_fields('pmra', 'pmdec')
from astroquery.gaia import Gaia
Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"  # Reselect Data Release 3, default

import scipy as sp
import scipy.ndimage
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
import astrasens_fitter as fitter


def get_dr2_id_from_tic(tic):
    '''
    Get Gaia parameters

    Returns
    -----------------------
    GaiaID, Gaia_mag
    '''
    # Get the Gaia sources
    result = Catalogs.query_object('TIC'+tic, radius=.005, catalog="TIC")

    IDs = result['ID'].data.data
    k = np.where(IDs == tic)[0][0]
    GAIAs = result['GAIA'].data.data
    Gaiamags = result['GAIAmag'].data.data

    GAIA_k = GAIAs[k]
    Gaiamag_k = Gaiamags[k]

    if GAIA_k == '':
        GAIA_k = np.nan
        sys.exit('ERROR: No Gaia DR2 ID found for this TIC number. If you have the Gaia DR3 ID try using the --gid option')
    return GAIA_k, Gaiamag_k

def dr3_from_dr2(dr2ID):
	query_dr3fromdr2 = "select dr3_source_id from gaiadr3.dr2_neighbourhood where dr2_source_id = "+dr2ID
	job = Gaia.launch_job(query=query_dr3fromdr2)
	dr3_ids = job.results['dr3_source_id'].value.data
	if len(dr3_ids) == 1:
		myid = dr3_ids[0]
	else:
		print("\t WARNING! There are more than one DR3 ids for this DR2 ID, assuming the brightest one...")
		gmags = np.zeros(len(dr3_ids))
		for ii,id in enumerate(dr3_ids):
			results = get_gaia_data_from_simbad(id)
			gmags[ii] = results['phot_g_mean_mag'].value.data
		brightest = np.argmin(gmags)
		myid = dr3_ids[brightest]

	return myid

def get_gaia_data_from_simbad(dr3ID):
	# simb = Simbad.query_object('Gaia DR2 '+dr2ID)
	# simbid = Simbad.query_objectids('Gaia DR2 '+dr2ID)
	# if simbid == None:
	#     print("ERROR: TIC not found in Simbad as Gaia DR2 "+str(dr2ID))
	# ids = np.array(simbid['ID'].data).astype(str)
	# myid = [id for id in ids if 'DR3' in id]
	# if len(myid) == 0:
	#     myid = [id for id in ids if 'DR2' in id]
	# myid = myid[0].split(' ')[2]

	myid = dr3ID #dr3_from_dr2(dr2ID)
	query2 = "SELECT \
				TOP 1 \
				source_id, ra, dec, pmra, pmdec, parallax, phot_g_mean_mag\
				FROM gaiadr3.gaia_source\
			    WHERE source_id = "+str(myid)+" \
			    "
	job = Gaia.launch_job_async(query2)
	gmag = job.get_results()['phot_g_mean_mag'].data[0]
	results = job.get_results()

	return results

def get_coord(tic):
    """
    Get TIC corrdinates

    Returns
    -------
    TIC number
    """
    try:
        catalog_data = Catalogs.query_object(objectname="TIC"+tic, catalog="TIC")
        ra = catalog_data[0]["ra"]
        dec = catalog_data[0]["dec"]
        # print(catalog_data.keys())
        # print(catalog_data[0]["GAIA"])
        return ra, dec
    except:
    	print("ERROR: TIC not found in Simbad")




def plotting(sources, target, popt, fake, myfwhm, center, sources2, args):

	# ==================================
	# PREPARING DATA
	# ==================================
	root = args.root
	night = args.night

	# Read the AstraLux image
	hdu = fits.open(root+'/11_REDUCED/'+night+'/'+args.image)

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

	data = hdu[0].data
	nx, ny = np.shape(data)

	# Load cdetected companions if the case:
	filecomp = 	root+'/22_ANALYSIS/'+night+'/DetectedSources/'+filename+'_Sources.csv'
	if os.path.isfile(filecomp):
		comps = np.atleast_1d(np.genfromtxt(filecomp,encoding='utf-8',dtype=None,names=True,delimiter=','))
		print(comps)
		ncomps =len(comps)
	else:
		ncomps=0

	# Load Sensitivity information
	t = np.load(root+'/22_ANALYSIS/'+night+'/Sensitivity/'+filename+'__Sensitivity.npz')
	detection = t['detection']
	dist_arr  = t['dist_arr']
	dmag_arr  = t['dmag_arr']
	sens = dist_arr*0.0

	maxdist  = np.min([ (nx-sources['xcentroid'][target])*0.02327-0.5,
						(ny-sources['ycentroid'][target])*0.02327-0.5,
						6. ])#3. # arcsec


	for i,dd in enumerate(dist_arr):
		sens[i] = np.interp( 0.7, np.cumsum(detection[i,:]), dmag_arr)

	if np.abs(sens[-2]-sens[-1]) > 1:
		dist_arr = dist_arr[:-1]
		sens = sens[:-1]

	# ==================================
	# CHECK GAIA
	# ==================================
	if 1:
		if "TOI" in objname: 
			TOIname = objname
		else:
			TOIname = None
		delta_ra,delta_dec,gid,ngaia = fitter.check_gaia(args,TOIname=objname)
		# print("\t --> Querying Gaia to look for the detected companions...")
		# # Read AstraLux targets
		# ast = np.genfromtxt('/Users/lillo_box/00_projects/11__HighResTESS/targets_table_astralux_TIC.csv',delimiter=',',encoding='utf-8',dtype=None,names=True)
		# try:
		# 	toi = np.abs(int(float(objname[3:])))
		# except:
		# 	toi = int(float(objname[4:]))
		# this = np.where(ast['TOI'] == toi)[0]
		# this = np.atleast_1d(this)
		# tic = str(ast['TIC'][this[0]])
		# ra,dec = get_coord(tic)
		# gaia_id, mag = get_dr2_id_from_tic(tic)
		# gaia_id = dr3_from_dr2(gaia_id)

		# coord = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame='fk5')
		# Gaia.ROW_LIMIT = -1
		# j = Gaia.cone_search_async(coord, radius=u.Quantity(5.0, u.arcsec))
		# gaiares = j.get_results()
		# gaiares.pprint()

		# # Position on CCD
		# ngaia = len(gaiares['source_id'].value.data) 
		# if ngaia > 1:
		# 	gaia_companions = {}
		# 	targ = np.where(gaiares['source_id'] == gaia_id)[0]
		# 	delta_ra,delta_dec,gid = [],[],[]
		# 	for i,row in enumerate(gaiares):
		# 		if row['source_id'] == gaia_id: continue
		# 		delta_ra.append(-1*(row['ra'].data - gaiares['ra'][targ].data)[0] * 3600)
		# 		delta_dec.append((row['dec'].data - gaiares['dec'][targ].data )[0] * 3600)
		# 		gid.append(row['source_id'])
		# 		print(row['source_id'])
		# 	delta_ra,delta_dec,gid = np.array(delta_ra), np.array(delta_dec), np.array(gid)



	# ======================================================================================
	# SUMMARY PLOT
	# ======================================================================================

	pxscale = 0.02327

	fig = plt.figure(figsize=(13,8))
	gs = gridspec.GridSpec(2,3, height_ratios=[1,1], width_ratios=[0.5,1,0.05])
	gs.update(left=0.05, right=0.98, bottom=0.15, top=0.93, wspace=0.25, hspace=0.5)

	# --------------------------------------------------------
	ax1 = plt.subplot(gs[0,0])  # RAW IMAGE
	# --------------------------------------------------------
	norm = ImageNormalize(stretch=SqrtStretch())
	plt.imshow(np.log(data), cmap='viridis', origin='lower',extent=[0,nx*0.02327,0.,ny*0.02327])
	plt.scatter(sources['xcentroid']*0.02327, sources['ycentroid']*0.02327, marker='o',s=100,facecolors='none',edgecolors='red',alpha=0.7)
	plt.scatter(sources['xcentroid'][target]*0.02327, sources['ycentroid'][target]*0.02327, marker='s',s=100,facecolors='none',edgecolors='red')
	circle1 = plt.Circle((sources['xcentroid'][target]*0.02327, sources['ycentroid'][target]*0.02327), 1, facecolor='none',ls=':', edgecolor='white')
	circle2 = plt.Circle((sources['xcentroid'][target]*0.02327, sources['ycentroid'][target]*0.02327), 2, facecolor='none',ls=':', edgecolor='white')
	ax1.add_patch(circle1)
	ax1.add_patch(circle2)

	plt.title('Full frame image')
	plt.xlabel('X (arcsec)')
	plt.ylabel('Y (arcsec)')

	# --------------------------------------------------------
	ax2 = plt.subplot(gs[1,0])   # RESIDUAL IMAGE
	# --------------------------------------------------------
	norm = ImageNormalize(stretch=SqrtStretch())
	residuals = data-fake
	# Block center
	residuals[int(sources['ycentroid'][target]-4):int(sources['ycentroid'][target]+4),
			  int(sources['xcentroid'][target]-4):int(sources['xcentroid'][target]+4)	] = np.nan
	# Plot image
	plt.imshow(residuals, cmap='viridis', origin='lower',norm=norm,
			   extent=[ -int(center[0])*0.02327,(nx-int(center[0]))*0.02327,
			   			-int(center[1])*0.02327,(ny-int(center[1]))*0.02327])
	# Plot companions and target
	if len(np.shape(sources2)) > 0:
		plt.scatter((sources2['xcentroid']-center[0])*0.02327, (sources2['ycentroid']-center[1])*0.02327, marker='o',s=100,facecolors='none',edgecolors='red',alpha=0.7)
	plt.scatter((sources['xcentroid'][target]-center[0])*0.02327, (sources['ycentroid'][target]-center[1])*0.02327, marker='s',s=100,facecolors='none',edgecolors='red')
	# Gaia data:
	if ngaia > 1:
		for _Dra, _Ddec, _gid in zip(delta_ra,delta_dec,gid):
			plt.scatter(_Dra, _Ddec, marker='D',s=100,facecolors='none',edgecolors='gold',alpha=0.7)
	# Circles and cosmetics
	circle1 = plt.Circle((0, 0), 1, facecolor='none',ls=':', edgecolor='white')
	circle2 = plt.Circle((0, 0), 2, facecolor='none',ls=':', edgecolor='white')
	ax2.add_patch(circle1)
	ax2.add_patch(circle2)
	plt.title('Residuals')
	plt.xlabel(r'$\Delta$X (arcsec)')
	plt.ylabel(r'$\Delta$Y (arcsec)')

	# --------------------------------------------------------
	ax3 = plt.subplot(gs[:,1])  # SENSITIVITY 
	# --------------------------------------------------------

	plt1 = ax3.scatter(dist_arr,sens, c='white',marker = 'o', s=40, edgecolors='none')
	#plt.plot(dist_arr,sens,c='white',zorder=-5,lw=2)
	try:
		f2 = interp1d(dist_arr,sens, kind='cubic')
		xnew = np.linspace(np.min(dist_arr), np.max(dist_arr), num=1000, endpoint=True)
		plt.plot(xnew,f2(xnew),c='white',zorder=-5,lw=2)
	except:
		plt.plot(dist_arr,sens,c='white')

	# Plot companions location:
	if ncomps > 0:
		for comp in comps:
			# sep2  = np.sqrt( ((s['xcentroid']-center[0])*0.02327)**2 + ((s['ycentroid']-center[1])*0.02327)**2)
			# dmag2 = s['mag']-sources[target]['mag']
			plt.scatter(comp['sep'],comp['dmag'],marker='*',c='gold',s=100)

	ax3.grid(True,ls=':',c='k')
	ax3.set_xlim([0,6])
	ax3.set_ylim([9.,0])
	ax3.set_xlabel(r' ') # Force this empty !
	ax3.set_ylabel(r'Contrast ($\Delta$m, mag)')
	ax3.set_xlabel('Angular separation (arcsec)')

	xlimup = np.floor(maxdist)+1
	ax3.text(xlimup*0.7, 1.2, 'AstraLux' + '\n' + 'Sensitivity curve, 5$\sigma$', color='black',
        	fontsize=14, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))


	x = np.linspace(0,6,100)
	y = np.linspace(9,0,100)
	X, Y = np.meshgrid(x,y)
	Z = Y/2.5
	Z1 = 10.**(-Z) *100.
	plt2 = ax3.imshow(Z1[::-1],extent=[0,6,9,0],zorder=-10,aspect='auto',
					  norm=LogNorm(vmin=Z1.max(), vmax=Z1.min()),cmap = 'winter_r')
	plt.xlim(0., xlimup)
	plt.title(objname+' ('+filter+')')


	# --------------------------------------------------------
	cbax = plt.subplot(gs[:,2])
	# --------------------------------------------------------
	cb = Colorbar(ax = cbax, mappable = plt2, orientation = 'vertical', ticklocation = 'left')
	cb.set_label('Contamination (%)', labelpad=0.1)

	plt.savefig(root+'/22_ANALYSIS/'+night+'/Summary_plots/'+filename+'__Summary.pdf')
	plt.close()



	# ======================================================================================
	# ZOOM TO THE RESIDUAL IMAGE
	# ======================================================================================

	fig = plt.figure(figsize=(8,8))
	gs = gridspec.GridSpec(1,1, height_ratios=[1], width_ratios=[1])
	gs.update(left=0.05, right=0.98, bottom=0.15, top=0.93, wspace=0.25, hspace=0.3)


	# --------------------------------------------------------
	ax2 = plt.subplot(gs[0,0])
	# --------------------------------------------------------
	norm = ImageNormalize(stretch=SqrtStretch())
	residuals = data-fake +100
	#residuals += 10.*np.median(residuals)
	#residuals[residuals<0] = np.nan
	white = 0
	residuals[int(sources['ycentroid'][target]-white):int(sources['ycentroid'][target]+white),
			  int(sources['xcentroid'][target]-white):int(sources['xcentroid'][target]+white)	] = np.nan

	sigma = [5,5]
	y = sp.ndimage.filters.gaussian_filter(residuals, sigma, mode='constant')
	norm = ImageNormalize(stretch=stretching.SinhStretch())  #SqrtStretch()
	residuals -= y

	#ind = np.unravel_index(np.argmax(residuals),np.shape(residuals))
	ind = np.where(residuals > 0.3*np.max(residuals))
	residuals[ind] = np.nan
	# ind = np.where(residuals < 0)
	# residuals[ind] = np.nan

	plt.imshow(residuals, cmap='viridis', origin='lower',norm=norm,
			   extent=[ -int(center[0])*0.02327,(nx-int(center[0]))*0.02327,
			   			-int(center[1])*0.02327,(ny-int(center[1]))*0.02327])

	if len(np.shape(sources2)) > 0:
		plt.scatter((sources2['xcentroid']-center[0])*0.02327, (sources2['ycentroid']-center[1])*0.02327, marker='o',s=100,facecolors='none',edgecolors='red',alpha=0.7)
	plt.scatter((sources['xcentroid'][target]-center[0])*0.02327, (sources['ycentroid'][target]-center[1])*0.02327, marker='s',s=100,facecolors='none',edgecolors='red')

	if ngaia > 1:
		for _Dra, _Ddec, _gid in zip(delta_ra,delta_dec,gid):
			plt.scatter(_Dra, _Ddec, marker='D',s=100,facecolors='none',edgecolors='gold',alpha=0.7)
			plt.text(_Dra+0.05, _Ddec+0.05, 'Gaia DR3 '+str(gid[0]), fontsize=10,c='k',alpha=0.8)

	circle1 = plt.Circle((0, 0), 0.5, facecolor='none',ls=':', edgecolor='white')
	circle2 = plt.Circle((0, 0), 1, facecolor='none',ls=':', edgecolor='white')
	circle3 = plt.Circle((0, 0), 2, facecolor='none',ls=':', edgecolor='white')
	ax2.add_patch(circle1)
	ax2.add_patch(circle2)
	ax2.add_patch(circle3)

	plt.title('Residuals')
	plt.xlabel(r'$\Delta$X (arcsec)')
	plt.ylabel(r'$\Delta$Y (arcsec)')
	plt.xlim(-2,2)
	plt.ylim(-2,2)

	plt.savefig(root+'/22_ANALYSIS/'+night+'/Summary_plots/'+filename+'__Residuals.pdf')
	plt.close()


	# ======================================================================================
	# ZOOM TO THE RESIDUAL IMAGE 2
	# ======================================================================================

	# --------------------
	fig = plt.figure(figsize=(8,8))
	gs = gridspec.GridSpec(1,1, height_ratios=[1], width_ratios=[1])
	gs.update(left=0.05, right=0.98, bottom=0.15, top=0.93, wspace=0.25, hspace=0.3)
	ax2 = plt.subplot(gs[0,0])

	norm = ImageNormalize(stretch=SqrtStretch())
	residuals = data-fake
	#residuals += 10.*np.median(residuals)
	#residuals[residuals<0] = np.nan
	square = 0
	residuals[int(sources['ycentroid'][target]-square):int(sources['ycentroid'][target]+square),
			  int(sources['xcentroid'][target]-square):int(sources['xcentroid'][target]+square)	] = np.nan

	plt.imshow(residuals, cmap='viridis', origin='lower',norm=norm,
			   extent=[ -int(center[0])*0.02327,(nx-int(center[0]))*0.02327,
			   			-int(center[1])*0.02327,(ny-int(center[1]))*0.02327])

	if len(np.shape(sources2)) > 0:
		plt.scatter((sources2['xcentroid']-center[0])*0.02327, (sources2['ycentroid']-center[1])*0.02327, marker='o',s=100,facecolors='none',edgecolors='red',alpha=0.7)
	plt.scatter((sources['xcentroid'][target]-center[0])*0.02327, (sources['ycentroid'][target]-center[1])*0.02327, marker='s',s=100,facecolors='none',edgecolors='red')

	if ngaia > 1:
		for _Dra, _Ddec, _gid in zip(delta_ra,delta_dec,gid):
			plt.scatter(_Dra, _Ddec, marker='D',s=100,facecolors='none',edgecolors='gold',alpha=0.7)
			plt.text(_Dra+0.05, _Ddec+0.05, 'Gaia DR3 '+str(gid[0]), fontsize=10,c='k',alpha=0.8)

	circle1 = plt.Circle((0, 0), 0.5, facecolor='none',ls=':', edgecolor='white')
	circle2 = plt.Circle((0, 0), 1, facecolor='none',ls=':', edgecolor='white')
	circle3 = plt.Circle((0, 0), 2, facecolor='none',ls=':', edgecolor='white')
	ax2.add_patch(circle1)
	ax2.add_patch(circle2)
	ax2.add_patch(circle3)
	plt.title('Residuals')
	plt.xlabel(r'$\Delta$X (arcsec)')
	plt.ylabel(r'$\Delta$Y (arcsec)')

	plt.axvline(0.0,ls=':',c='k')
	plt.axhline(0.0,ls=':',c='k')

	plt.xlim(-2,2)
	plt.ylim(-2,2)

	plt.savefig(root+'/22_ANALYSIS/'+night+'/Summary_plots/'+filename+'__Residuals2.pdf')
	plt.close()
