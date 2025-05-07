import numpy as np
from astropy.io import fits
from sys import platform
from numpy import unravel_index
from matplotlib.colors import LogNorm
import matplotlib
from astropy.io import ascii
from pylab import *
import argparse
from astropy.io import fits
import time
from scipy.optimize import curve_fit
from astropy.io import ascii
import os
from astropy.table import Table, Column, MaskedColumn


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="npz file of the contrast curve")
    parser.add_argument("root", help="Root folder")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    """
        Convert the npz file of the AstraLux CONTRAST curve into an ascii file
    """

    args = cli()

    # Read AstraLux
    t = np.load(args.root+'/'+args.file)
    detection = t['detection']
    sep_Astr1  = t['dist_arr']
    dmag_arr  = t['dmag_arr']
    sens1 = sep_Astr1*0.0
    for i,dd in enumerate(sens1):
        sens1[i] = np.interp( 0.7, np.cumsum(detection[i,:]), dmag_arr)
    data = Table([sep_Astr1,sens1],names=['#Separation_arcsec','5s-contrast'])
    prefix = args.file.split('.')[0]
    ascii.write(data, args.root+'/'+prefix+'.txt',overwrite=True)
