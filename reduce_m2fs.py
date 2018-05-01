"""
File name: m2fs_biassubtract.py
Author: Alex Ji (based on code by T.T. Hansen and J.D. Simon)
Date Created: 4/24/2018
Date last modified: 4/24/2018
"""

#if __name__=="__main__":
#    config = read_config()
    
import numpy as np
import glob, os, sys
from importlib import reload

from m2fs_utils import mrdfits, read_fits_two, write_fits_two
from m2fs_utils import m2fs_4amp

import m2fs_reduction_pipeline; reload(m2fs_reduction_pipeline)

def m2fs_biastrim(datadir):
    inlist = glob.glob(datadir+"/*c1.fits")
    nfiles = len(inlist)
    for i in range(nfiles):
        infile = inlist[i].split("c1")[0]
        m2fs_4amp(infile)

def m2fs_darksub(rb, rwd, twd, mdarkfname):
    inlist = glob.glob("{}/{}????.fits".format(rwd,rb))
    def make_out_name(fname):
        name = os.path.basename(fname)
        assert name.endswith(".fits")
        return os.path.join(twd,name[:-5]+"d.fits")
    outlist = map(make_out_name, inlist)
    
    dark, darkerr, darkheader = read_fits_two(mdarkfname)
    
    print("Running dark subtraction for {} files ({} arm)".format(
            len(inlist), rb))
    for infile, outfile in zip(inlist, outlist):
        m2fs_reduction_pipeline.m2fs_subtract_one_dark(infile, outfile, dark, darkerr, darkheader)

if __name__ == "__main__":
    rb = "r"
    rwd = "../raw"
    twd = "../reduction_files"
    
    mdarkfname = "{}/mdark_{}.fits".format(twd,rb)
    mflatfname = "{}/mflat_{}.fits".format(twd,rb)
    
    ## Bias subtraction and trimming
    #m2fs_biastrim(rwd)
    ## Make master darks
    darks = ["{}/{}{:04}.fits".format(rwd,rb,i) for i in [1230, 1231, 1232]]
    m2fs_reduction_pipeline.m2fs_make_master_dark(darks, mdarkfname)
    ## Dark subtraction
    m2fs_darksub(rb,rwd,twd,mdarkfname)
    
    ## Make master flat
    filenames = ["{}/{}{:04}d.fits".format(twd,rb,i) for i in [1417,1423,1435,1568,1586]]
    Nflat = len(filenames)
    flat, h = mrdfits(filenames[0], 0)
    Nx, Ny = flat.shape
    
    master_flat = np.empty((Nflat, Nx, Ny))
    master_flaterr = np.empty((Nflat, Nx, Ny))
    for k,fname in enumerate(filenames):
        master_flat[k], master_flaterr[k], flathead = read_fits_two(fname)
    master_flat = np.median(master_flat, axis=0)
    master_flaterr = np.median(master_flaterr, axis=0)
    
    outfname = mflatfname
    write_fits_two(outfname, master_flat.T, master_flaterr.T, h)
    print("Created master flat and wrote to {}".format(outfname))

def terese():
    ## (1) combine image from the four amplifiers
    ## do bias subtraction using the overscan region
    th_m2fs_reduce(run1=1,run2=0)
    ## produces pbbb.fits blue 4 amplifiers merged
    
    ## (2) make master dark image
    mdark()
    ## subtract dark current
    th_m2fs_reduce(run1=0,run2=1)
    
    ## (3) combine twilight and quartz flats
    th_m2fs_mflat()
    
    ## (4) remove cosmic from science frames
    th_m2fs_crbye()
    
    ## (5) trace orders
    ## Note: check trace with atv, for half orders the trace can be bad
    th_m2fs_trace()
    
    ## (6) flat correct science frames
    th_m2fs_flat()
    
    ## (7) combine science frames from same night
    th_m2fs_imcombine()
    
    ## (8) use IRAF (apfind, aptrace, apall, identify) to extracte middle 4
    ## orders on one of the blue and red ThAr frames and find wavelength solution.
    ## Create list of pixel and lambda values for each of the four orders.
    fit_middle_orders()
    
    ## (9) use mpfit to create legendre polynomial fit to the pixel and wavelength values
    ## based on cross correlation with order examined in IRAF.
    ## Then this program extracts the middle 4 orders and cross-correlates with rest to find offset
    ## then is used to find wavelength solution using 
    ## Note: change pixel-lambda list for the four order and the template set of orders to the right order nr
    th_m2fs_wave_mpfit()
    
    ## (10) sky subtraction
    ## create skyfib list from sky fibers in red and blue CCD
    th_m2fs_skysub()

    ## (11) extract
    ## Extracts each order into a single spectrum
    th_m2fs_extract()
    
    
