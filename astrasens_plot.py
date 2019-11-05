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
import circles


def plotting(sources, target, popt, fake, myfwhm, center, sources2, args):

	# ==================================
	# PLOTS
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
		idobs = file.split('_')[3]
		filter = file.split('_')[4]
	else:
		idobs = ''
		filter = file.split('_')[3]
	
	filename = file[14:]+'_'+rate
	
	data = hdu[0].data
	nx, ny = np.shape(data)
	
	# Load Sensitivity information
	t = np.load(root+'/22_ANALYSIS/'+night+'/Sensitivity/'+filename+'__Sensitivity.npz')
	detection = t['detection']
	dist_arr  = t['dist_arr']
	dmag_arr  = t['dmag_arr']
	sens = dist_arr*0.0
	print dist_arr

	maxdist  = np.min([ (nx-sources['xcentroid'][target])*0.02327-0.5, 
						(ny-sources['ycentroid'][target])*0.02327-0.5, 
						6. ])#3. # arcsec

	
	for i,dd in enumerate(dist_arr):
		sens[i] = np.interp( 0.7, np.cumsum(detection[i,:]), dmag_arr)
	
	if np.abs(sens[-2]-sens[-1]) > 1: 
		dist_arr = dist_arr[:-1]
		sens = sens[:-1]
	
	fig = plt.figure(figsize=(13,8))
	gs = gridspec.GridSpec(2,3, height_ratios=[1,1], width_ratios=[0.5,1,0.05])
	gs.update(left=0.05, right=0.98, bottom=0.15, top=0.93, wspace=0.25, hspace=0.3)

	# --------------------------------------------------------
	ax1 = plt.subplot(gs[0,0]) 
	# --------------------------------------------------------
	norm = ImageNormalize(stretch=SqrtStretch())
	plt.imshow(np.log(data), cmap='viridis', origin='lower',extent=[0,nx*0.02327,0.,ny*0.02327])
	plt.scatter(sources['xcentroid']*0.02327, sources['ycentroid']*0.02327, marker='o',s=100,facecolors='none',edgecolors='red',alpha=0.7)
	plt.scatter(sources['xcentroid'][target]*0.02327, sources['ycentroid'][target]*0.02327, marker='s',s=100,facecolors='none',edgecolors='red')
	plt.title('Full frame image')
	plt.xlabel('X (arcsec)')
	plt.ylabel('Y (arcsec)')

	# --------------------------------------------------------
	ax2 = plt.subplot(gs[1,0]) 
	# --------------------------------------------------------
	norm = ImageNormalize(stretch=SqrtStretch())
	residuals = data-fake
	#residuals += 10.*np.median(residuals)
	#residuals[residuals<0] = np.nan	
	residuals[int(sources['ycentroid'][target]-4):int(sources['ycentroid'][target]+4),
			  int(sources['xcentroid'][target]-4):int(sources['xcentroid'][target]+4)	] = np.nan

	plt.imshow(residuals, cmap='viridis', origin='lower',norm=norm,
			   extent=[ -int(center[0])*0.02327,(nx-int(center[0]))*0.02327,
			   			-int(center[1])*0.02327,(ny-int(center[1]))*0.02327])
	
	if len(sources2) > 0:
		plt.scatter((sources2['xcentroid']-center[0])*0.02327, (sources2['ycentroid']-center[1])*0.02327, marker='o',s=100,facecolors='none',edgecolors='red',alpha=0.7)
	plt.scatter((sources['xcentroid'][target]-center[0])*0.02327, (sources['ycentroid'][target]-center[1])*0.02327, marker='s',s=100,facecolors='none',edgecolors='red')
	circles.circles(0, 0, 1, 'none',ls=':', edgecolor='white')
	circles.circles(0, 0, 2, 'none',ls=':', edgecolor='white')
	plt.title('Residuals')
	plt.xlabel(r'$\Delta$X (arcsec)')
	plt.ylabel(r'$\Delta$Y (arcsec)')
	
	# --------------------------------------------------------
	ax3 = plt.subplot(gs[:,1]) 
	# --------------------------------------------------------
	
	plt1 = ax3.scatter(dist_arr,sens, c='white',marker = 'o', s=40, edgecolors='none')
	#plt.plot(dist_arr,sens,c='white',zorder=-5,lw=2)
	f2 = interp1d(dist_arr,sens, kind='cubic')
	xnew = np.linspace(np.min(dist_arr), np.max(dist_arr), num=1000, endpoint=True)
	plt.plot(xnew,f2(xnew),c='white',zorder=-5,lw=2)
	
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
	cb.set_label('Contamination (\%)', labelpad=0.)

	plt.savefig(root+'/22_ANALYSIS/'+night+'/Summary_plots/'+filename+'__Summary.pdf')
	plt.close()
				
