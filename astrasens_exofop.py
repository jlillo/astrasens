#!/usr/bin/env python

'''
--- astrasens_exofop.py ---

	Purpose
	--------
	Create exofop summary table to upload

	Inputs and Options
	------------------
	
	Example:
	--------
	python astrasens_exofop.py image 

	Version Control:
	----------------
	2018/12/05   jlillobox First version 

'''
# ======================================
# Imports
# ======================================

import numpy as np
import emcee
import scipy.optimize as op
from math import *
import get_xx
from astropy.io import ascii
import emcee
import scipy.optimize as op
import sys
from astropy.table import Table, Column
import math
#import reveal_prepare as prepare
import astrasens_fitter as fitter
import astrasens_plot  as plotting
import multiprocessing
import time
import argparse
import os
from termcolor import colored
import glob
import ntpath

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="Root folder")
    parser.add_argument("night", help="Night folder")
    parser.add_argument("-V", "--VERBOSE", help="VERBOSE", action="store_true")
    parser.add_argument("-P", "--PLOTS", help="Plots only", action="store_true")
    args = parser.parse_args()
    return args

# ======================================
# 	        MAIN
# ======================================

if __name__ == "__main__":

	args = cli()
	root = args.root
	images = glob.glob(args.root+'/11_REDUCED/'+args.night+'/*0100*TOI*.fits')
	
	
	f = open('obs-imaging-20'+args.night+'-000.txt', 'w')
	
	for image in images:
		filename = ntpath.basename(image)
		# Get information from image name and header
		file = os.path.splitext(filename)[0]
		print file,len(file.split('_'))
		objname = file.split('_')[2]
		objname.replace('-','')
		rate = file.split('_')[1]
		if len(file.split('_')) == 7: # For cases whith more than one obs per night (e.g., TOI-XXXX_1)
			idobs = file.split('_')[3]
			filter = file.split('_')[3]
		else:
			idobs = ''
			filter = file.split('_')[3]
		
		fileID = file[14:]+'_'+rate


		# Load Sensitivity information
		if os.path.isfile(args.root+'/22_ANALYSIS/'+args.night+'/Sensitivity/'+fileID+'__Sensitivity.npz'):
			t = np.load(root+'/22_ANALYSIS/'+args.night+'/Sensitivity/'+fileID+'__Sensitivity.npz')
			detection = t['detection']
			dist_arr  = t['dist_arr']
			dmag_arr  = t['dmag_arr']
			sens = dist_arr*0.0

			for i,dd in enumerate(dist_arr):
				sens[i] = np.interp( 0.7, np.cumsum(detection[i,:]), dmag_arr)
		
			contrast = str(np.round(np.interp(1.0, dist_arr, sens),1))
		else:
			contrast = '--'
		
		nightformat = '20'+args.night[0:2]+'-'+args.night[2:4]+'-'+args.night[4:6]
		f.write(objname+'|2.2m@CAHA|2.2m|AstraLux|'+filter+'|909.7|137|nm|Lucky|0.02327||'+contrast+'|1|'+nightformat+'|20'+args.night+'_lillobox_astralux_00000|| \n')
		print objname

	f.close()


# \ Tel = Telescope*
# \ TelSize = Telescope size (meters)
# \ Inst = Instrument*
# \ Filter = Filter name*
# \ FiltCent = Filter central wavelength*
# \ FiltWidth = Filter width*
# \ FiltUnits = Filter units -- nm, Angstroms, or microns*
# \ ImageType = AO, Speckle, Lucky, Seeing-Limited, or Other*
# \ Pixscale = Pixel scale (arcsec)*
# \ PSF = Estimated PSF (arcsec)
# \ Contrast_mag = Estimated contrast magnitude
# \ Contrast_sep = Estimated contrast separation (arcsec)
# \ Obsdate = Observation date (UT)* -- format YYYY-MM-DD (hh:mm:ss optional)
# \ Tag = data tag number or name (e.g. YYYYMMDD_username_description_nnnnn)*
# \ Group = group name
# \ Notes = notes about the observation
# \
# \ Example data (header line is not required):
# 
# Target|Tel|TelSize|Inst|Filter|FiltCent|FiltWidth|FiltUnits|ImageType|Pixscale|PSF|Contrast_mag|Contrast_sep|Obsdate|Tag|Group|Notes




