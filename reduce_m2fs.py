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
from scipy import optimize, signal, ndimage, special, linalg

from astropy.io import fits

from m2fs_utils import mrdfits, read_fits_two, write_fits_two, m2fs_load_files_two
from m2fs_utils import gaussfit, jds_poly_reject, m2fs_extract1d
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
        fibermapfname = "fibermap_red.txt"
        x_begin = 750
        x_end = 1300
    elif rb == "b":
        expected_fibers = 144
        fibermapfname = "fibermap_blue.txt"
        x_begin = 900
        x_end = 1300
    else:
        raise ValueError(rb)
    rwd = "../raw"
    twd = "../reduction_files"
    
    mdarkfname = "{}/mdark_{}.fits".format(twd,rb)
    mflatfname = "{}/mflat_{}.fits".format(twd,rb)
    tracefname = "{}/mflat_{}_tracecoeff.txt".format(twd,rb)
    tracestdfname = "{}/mflat_{}_tracestdcoeff.txt".format(twd,rb)
    n1_mflatfname = "{}/n1_mflat_{}.fits".format(twd,rb)
    n2_mflatfname = "{}/n2_mflat_{}.fits".format(twd,rb)
    
    darknums = [1230,1231,1232]
    n1_flatnum = [1401,1417,1423,1435]
    n2_flatnum = [1565,1568,1586]
    flatnums = n1_flatnum + n2_flatnum
    n1_scinum = [1419,1420,1424,1427,1428,1431,1432]
    n2_scinum = [1570,1571,1574,1575,1578,1579,1582,1583]
    scinums = n1_scinum + n2_scinum

    darks = ["{}/{}{:04}.fits".format(rwd,rb,i) for i in darknums]
    flats = ["{}/{}{:04}d.fits".format(twd,rb,i) for i in flatnums]
    sciences = ["{}/{}{:04}d.fits".format(twd,rb,i) for i in scinums]
    scicrrs = ["{}/{}{:04}dcrr.fits".format(twd,rb,i) for i in scinums]
    scicrrfs = ["{}/{}{:04}dcrrf.fits".format(twd,rb,i) for i in scinums]
    
    n1_flats = ["{}/{}{:04}d.fits".format(twd,rb,i) for i in n1_flatnum]
    n2_flats = ["{}/{}{:04}d.fits".format(twd,rb,i) for i in n2_flatnum]
    n1_scis = ["{}/{}{:04}d.fits".format(twd,rb,i) for i in n1_scinum]
    n2_scis = ["{}/{}{:04}d.fits".format(twd,rb,i) for i in n2_scinum]
    n1_scicrrs = ["{}/{}{:04}dcrr.fits".format(twd,rb,i) for i in n1_scinum]
    n2_scicrrs = ["{}/{}{:04}dcrr.fits".format(twd,rb,i) for i in n2_scinum]
    n1_scicrrfs = ["{}/{}{:04}dcrrf.fits".format(twd,rb,i) for i in n1_scinum]
    n2_scicrrfs = ["{}/{}{:04}dcrrf.fits".format(twd,rb,i) for i in n2_scinum]
    
    ## Bias subtraction and trimming
    #m2fs_biastrim(rwd)
    ## Make master darks and subtract
    #m2fs_reduction_pipeline.m2fs_make_master_dark(darks, mdarkfname)
    #m2fs_darksub(rb,rwd,twd,mdarkfname)
    ## Make master flats
    #m2fs_reduction_pipeline.m2fs_make_master_flat(flats, mflatfname)
    #m2fs_reduction_pipeline.m2fs_make_master_flat(n1_flats, n1_mflatfname)
    #m2fs_reduction_pipeline.m2fs_make_master_flat(n2_flats, n2_mflatfname)
    ## Trace Flat
    #m2fs_reduction_pipeline.m2fs_trace_orders(mflatfname, expected_fibers)
    #m2fs_reduction_pipeline.m2fs_trace_orders(n1_mflatfname, expected_fibers)
    #m2fs_reduction_pipeline.m2fs_trace_orders(n2_mflatfname, expected_fibers)
    ## Remove cosmic rays from science frames, using all frames from a given night
    #m2fs_reduction_pipeline.m2fs_remove_cosmics(n1_scis)
    #m2fs_reduction_pipeline.m2fs_remove_cosmics(n2_scis)
    ## Apply flat to science frames
    #m2fs_reduction_pipeline.m2fs_flat(scicrrs, mflatfname, tracefname, fibermapfname,
    #                                  x_begin=x_begin, x_end=x_end)
    # TODO why can't I just divide one by the other? Because there's lots of 0s.
    # Without a milky flat, it seems best to just extract the object without applying the flat.
    # Then after also extracting the flat, then divide the 1D spectra.
    # There's no need to do this in 2D!

    # Find extraction profile from flat
    m2fs_reduction_pipeline.m2fs_ghlb_profile(mflatfname, tracefname, tracestdfname, fibermapfname,
                                              x_begin, x_end)
    
    ## Coadd
    #m2fs_reduction_pipeline.m2fs_imcombine(scicrrfs, os.path.join(twd, "science_{}_dcf.fits".format(rb)))
    # TODO the better way is to simultaneously fit this out of the individual frames
    #dcfws is the final one
    
def tmp():
    ## wavelength calibration
    yaper=7.     #                                 ; aperture for collapsing 2D arc spectrum to 1D
    degree=4     #                               ; degree of wavelength solution
    nthresh=10.0 #;orig=10                                  ; N times above the stddev of the 1D spectrum
    saturation=30000.0 #                           ; saturation level to remove saturated lines
    dx=10        #                                 ; width of x window for gaussian fit to find line center
    px=17.0      #                                 ; pixrange for picking out peaks with known lambda values, change this is central wavelenght not right, ie because outer peaks are not found
    
    # read in fibermap and trace
    # read arc frame
    nfib = len(tracecoefs)
    wavecoefs = np.zeros((degree, nfib))
    rms = np.zeros(nfib)
    # read in lines <-> pixels done manually from IRAF
    
    # extract one set of orders to use for cross correlation
    # Use biweight mean of things in the aperture
    
    # for each fiber:
    #   extract the arc spectrum to 1D
    #   <apply a mask if needed>
    #   remove continuum
    #   find peaks
    #   cross correlate to get offset from original orders
    #   match peaks to pix list/lambda
    #   fit wavelength solution, remove bad lines manually
    

def terese():
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
    # TODO object profile?
    # TODO extract from the summed 2D spectrum, or simultaneous extract individual 2D spectra?
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
    
