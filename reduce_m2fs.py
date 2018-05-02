"""
File name: reduce_m2fs.py
Author: Alex Ji (based on code by T.T. Hansen and J.D. Simon)
Date Created: 4/24/2018
Date last modified: 5/1/2018
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
    
import numpy as np
import glob, os, sys, time
from importlib import reload

from astropy.io import fits

from m2fs_utils import mrdfits, read_fits_two, write_fits_two, m2fs_load_files_two
from m2fs_utils import gaussfit
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
    
    start = time.time()
    print("Running dark subtraction for {} files ({} arm)".format(
            len(inlist), rb))
    for infile, outfile in zip(inlist, outlist):
        m2fs_reduction_pipeline.m2fs_subtract_one_dark(infile, outfile, dark, darkerr, darkheader)
    print("Took {:.1f}s".format(time.time()-start))

if __name__ == "__main__":
    rb = "b"
    if rb == "r":
        expected_fibers = 112
    elif rb == "b":
        expected_fibers = 144
    else:
        raise ValueError(rb)
    rwd = "../raw"
    twd = "../reduction_files"
    
    mdarkfname = "{}/mdark_{}.fits".format(twd,rb)
    mflatfname = "{}/mflat_{}.fits".format(twd,rb)
    n1_mflatfname = "{}/n1_mflat_{}.fits".format(twd,rb)
    n2_mflatfname = "{}/n2_mflat_{}.fits".format(twd,rb)
    
    darknums = [1230, 1231, 1232]
    n1_flatnum = [1417,1423,1435]
    n2_flatnum = [1568,1586]
    flatnums = n1_flatnum + n2_flatnum
    n1_scinum = [1419,1420,1424,1427,1428,1431,1432]
    n2_scinum = [1570,1571,1574,1575,1578,1579,1582,1583]
    scinums = n1_scinum + n2_scinum

    darks = ["{}/{}{:04}.fits".format(rwd,rb,i) for i in darknums]
    flats = ["{}/{}{:04}d.fits".format(twd,rb,i) for i in flatnums]
    sciences = ["{}/{}{:04}d.fits".format(twd,rb,i) for i in scinums]
    scicrrs = ["{}/{}{:04}dcrr.fits".format(twd,rb,i) for i in scinums]
    
    n1_flats = ["{}/{}{:04}d.fits".format(twd,rb,i) for i in n1_flatnum]
    n2_flats = ["{}/{}{:04}d.fits".format(twd,rb,i) for i in n2_flatnum]
    n1_scis = ["{}/{}{:04}d.fits".format(twd,rb,i) for i in n1_scinum]
    n2_scis = ["{}/{}{:04}d.fits".format(twd,rb,i) for i in n2_scinum]
    n1_scicrrs = ["{}/{}{:04}dcrr.fits".format(twd,rb,i) for i in n1_scinum]
    n2_scicrrs = ["{}/{}{:04}dcrr.fits".format(twd,rb,i) for i in n2_scinum]
    
    ## Bias subtraction and trimming
    #m2fs_biastrim(rwd)
    ## Make master darks and subtract
    #m2fs_reduction_pipeline.m2fs_make_master_dark(darks, mdarkfname)
    #m2fs_darksub(rb,rwd,twd,mdarkfname)
    ## Make master flats
    #m2fs_reduction_pipeline.m2fs_make_master_flat(flats, mflatfname)
    #m2fs_reduction_pipeline.m2fs_make_master_flat(n1_flats, n1_mflatfname)
    #m2fs_reduction_pipeline.m2fs_make_master_flat(n2_flats, n2_mflatfname)
    ## Remove cosmic rays from science frames, using all frames from a given night
    #m2fs_reduction_pipeline.m2fs_remove_cosmics(n1_scis)
    #m2fs_reduction_pipeline.m2fs_remove_cosmics(n2_scis)
    ## Trace Flat
    #m2fs_reduction_pipeline.m2fs_trace_orders(mflatfname, expected_fibers)
    
def terese():
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
    
    
    # Dan Kelson's MIKE steps:
    # 1. bias/trim/reorient
    # 2. apply flat: flat2d
    # 3. apply y distortion (stage-ydist-copy): copyrect from slitred -> lampf
    # 4. trace order edges and apply to lamp/obj (stage-orders-copy): copyslit
    # 5. divide out 2d image of slit function (stage-deslitfn): mikeflat1d
    # 6. get x distortion (stage-xdist): (apply to lampfsb)
    # 7. copy x distortion (stage-xdist-copy): copyrect 
    # 8. wave calib (stage-wdist): mikeMatchOrders, mikeFindLines, mikeMatchLamps
    # 9. copy wave (stage-wdist-copy)
    # 10. generate sky apertures (stage-skyaps): mkorderaps
    # 11. find sky bg (stage-skyrect): skyrect
    # 12. sky subtract (stage-skysub): specfso = specfs - specfsm
    # 13. find object positions (stage-objaps): mkorderaps -> spec<rb>.aps
    # 14. mask bad pixels (stage-mkmask): flat > 0.25
    # 15. extract (stage-extractall): GHLBExtract
    #       input: lampfs, flat blaze<rb>, mask, header, fs.aps spec<rb>fso.aps
    
