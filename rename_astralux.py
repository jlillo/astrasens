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
import glob
import argparse
import shutil

def cli():
    """command line inputs

    Get parameters from command line

    Returns
    -------
    Arguments passed by command line
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to folder")
    args = parser.parse_args()
    return args

# ======================================
# 	        MAIN
# ======================================

if __name__ == "__main__":

    args = cli()

    files = np.array(glob.glob(os.path.join(args.path,"ast*.fits")))
    prefix = []
    for file in files:
        prefix.append(file[:-6])
    prefix = np.array(prefix)
    unique = np.unique(prefix)

    rates = ['']

    f = open('rename.bash','w')

    for obj in unique:
        this = np.where(prefix == obj)[0]
        this_files = files[this]
        selectio,filename,origfile = [],[],[]
        nhighpix = np.zeros(len(this_files))
        for ff,file in enumerate(this_files):
            a = fits.open(file)
            data = a[0].data
            header = a[0].header
            nhighpix[ff] = len(np.where(data > 0.9*np.max(data))[0])
            selectio.append(header['SELECTIO'])
            filename.append(header['FILENAME'])
            origfile.append(header['ORIGFILE'])

        selectio,filename,origfile = np.array(selectio),np.array(filename),np.array(origfile)
        print(obj)
        print(selectio,filename,origfile)


        uselec = np.unique(selectio)
        Nfiles = 0
        for sel in uselec:
            this = np.where(selectio == sel)[0]
            _filename = filename[this]
            _nhighpix = nhighpix[this]
            _this_files = this_files[this]

            min_nhighpix = np.where(_nhighpix == np.min(_nhighpix))[0]
            my_filename = _filename[min_nhighpix]
            my_this_files = _this_files[min_nhighpix]
            print(my_filename)
            print(my_this_files[0])
            # print(my_this_files, my_filename,_nhighpix,_nhighpix[min_nhighpix])
            # os.rename(my_this_files[0], os.path.join(args.path,my_filename[0]))
            f.write('mv '+my_this_files[0]+' '+os.path.join(args.path,my_filename[0])+'\n')
            Nfiles += 1
        # print('%20a %20i' % (header['ORIGFILE'],Nfiles))
        # sys.exit()
    f.close()

    os.mkdir(os.path.join(args.path,'trash'))
    # for file in glob.glob(os.path.join(args.path,"ast*.fits")):
        # shutil.move(file,os.path.join(args.path,'trash',os.path.basename(file)))
