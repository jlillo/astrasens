#!/usr/bin/env python

'''
--- astrasens_run.py ---

	Purpose
	--------
	Get sensitivity limits and create summary plots

	Inputs and Options
	------------------


	Example:
	--------
	python astrasens_run.py image

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


def run(args):
	'''run

	Main function to run the different scripts with different options

	Parameter
	---------
	image		image name

	Returns
	-------
	none

	'''
	ana_fold   = args.root+'/22_ANALYSIS/'
	night_fold = args.root+'/22_ANALYSIS/'+args.night+'/'
	sens_fold  = args.root+'/22_ANALYSIS/'+args.night+'/Sensitivity/'
	sour_fold  = args.root+'/22_ANALYSIS/'+args.night+'/DetectedSources/'
	summ_fold  = args.root+'/22_ANALYSIS/'+args.night+'/Summary_plots/'

	if (os.path.isdir(ana_fold) == False): os.mkdir(ana_fold)
	if (os.path.isdir(night_fold) == False): os.mkdir(night_fold)
	if (os.path.isdir(sens_fold) == False): os.mkdir(sens_fold)
	if (os.path.isdir(sour_fold) == False): os.mkdir(sour_fold)
	if (os.path.isdir(summ_fold) == False): os.mkdir(summ_fold)

	if args.PLOTS == True:
		sources, target, popt, fake, myfwhm, center, sources2 = fitter.sources(args)
		plotting.plotting(sources, target, popt, fake, myfwhm, center, sources2, args)
	else:
		# ===== Find sources in the image
		print("\n")
		print("+++++++++++++++++++++++++")
		print("Source identification...")
		print("+++++++++++++++++++++++++")
		filename = fitter.get_filename(args)
		outfile = args.root+'/22_ANALYSIS/'+args.night+'/DetectedSources/'+filename+'__Sources.npz'
		if os.path.isfile(outfile) == True and args.FORCEDET == False:
			print("\t --> File *Sources.nzp found, sources recovered. Use --FORCEDET to force re-analysis.")
			results = np.load(outfile)
			sources = results['sources']
			target = results['target']
			popt = results['popt']
			fake = results['fake']
			myfwhm = results['myfwhm']
			center = results['center']
			sources2 = results['sources2']
		else:
			sources, target, popt, fake, myfwhm, center, sources2 = fitter.sources(args)


		# ===== Sensitivity
		print("\n")
		print("+++++++++++++++++++++++++")
		print("Sensitivity curve... ")
		print("+++++++++++++++++++++++++")
		# file = os.path.splitext(args.image)[0]
		# rate = file.split('_')[1]
		# filename = file[14:]+'_'+rate
		sensitivity_file = args.root+'/22_ANALYSIS/'+args.night+'/Sensitivity/'+filename+'__Sensitivity.npz'
		if os.path.isfile(sensitivity_file) and args.FORCE is not True:
			print("\t --> File found, sensitivity curve not re-calculated. Use --FORCE to force overwrite")
		else:
			fitter.sensitivity(sources, target, popt, fake, myfwhm, args)

		# ===== Plots
		print("\n")
		print("+++++++++++++++++++++++++")
		print("Plotting... ")
		print("+++++++++++++++++++++++++")
		plotting.plotting(sources, target, popt, fake, myfwhm, center, sources2, args)



def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Image name")
    parser.add_argument("root", help="Root folder")
    parser.add_argument("night", help="Night folder")
    parser.add_argument("-G", "--GDR3", help="Gaia DR3 id of the observed source", default=None)
    parser.add_argument("-V", "--VERBOSE", help="VERBOSE", action="store_true")
    parser.add_argument("-P", "--PLOTS", help="Plots only", action="store_true")
    parser.add_argument("-F", "--FORCE", help="Force re-calculation of sensitivity curve", action="store_true")
    parser.add_argument("-FD", "--FORCEDET", help="Force detection of source companions", action="store_true")
    parser.add_argument("-I", "--IPAC", help="Format output of detected sources", action="store_true")
    parser.add_argument("-W", "--WINDOW", help="Maximum distance to consider", default=None,type=float)
    parser.add_argument("-T", "--TIC", help="TIC id of the object if known", default=None,type=int)
    args = parser.parse_args()
    return args



# ======================================
# 	        MAIN
# ======================================

if __name__ == "__main__":

	print(        "............................................")
	print(colored("           _                                ", "light_yellow"))
	print(colored("  __ _ ___| |_ _ __ __ _ ___  ___ _ __  ___ ", "light_yellow"))
	print(colored(" / _` / __| __| '__/ _` / __|/ _ \ '_ \/ __|", "light_yellow"))
	print(colored("| (_| \__ \ |_| | | (_| \__ \  __/ | | \__ \ ", "light_yellow"))
	print(colored(" \__,_|___/\__|_|  \__,_|___/\___|_| |_|___/", "light_yellow"))
	print(        "               by @jlillobox                ")
	print(        "............................................")
	print('\n')

	args = cli()

	if args.image == 'all':
		_images = glob.glob(os.path.join(args.root,'11_REDUCED',args.night,'TDRIZZLE*0100*.fits'))
		fok = open(os.path.join(args.root,'files_completed.lis'),'w')
		fnotok = open(os.path.join(args.root,'files_error.lis'),'w')
		for i in _images:
			image = os.path.basename(i)
			print(colored("=======================================================", "light_blue"))
			print(colored("Running image "+image, "light_blue"))
			print(colored("=======================================================", "light_blue"))
			args.image = image
			try:
				run(args)
				fok.writelines(image+'\n')
				print(colored(image +"...ok \n", 'green'))
			except:
				fnotok.writelines(image+'\n')
				print(colored("ERROR:" +image +" could not be processed \n", 'red'))
		fok.close()
		fnotok.close()

	elif args.image.endswith('.lis'):
		images = np.genfromtxt(args.image, dtype=None, encoding='utf-8')
		for image in images:
			print(colored("=======================================================", "light_blue"))
			print(colored("Running image "+image, "light_blue"))
			print(colored("=======================================================", "light_blue"))
			args.image = image
			try:
				run(args)
				print(colored(image +"...ok \n", 'green'))
			except:
				print(colored("ERROR:" +image +" could not be processed \n", 'red'))
	else:
		print(colored("=======================================================", "light_blue"))
		print(colored("Running image "+args.image, "light_blue"))
		print(colored("=======================================================", "light_blue"))
		run(args)
		print('\n')
		print(colored(args.image +"...ok \n", 'green'))
