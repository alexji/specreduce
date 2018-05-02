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
    rb = "r"
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

    # Initialize
    fname, tracefname, tracestdfname, fibermapfname = mflatfname, tracefname, tracestdfname, fibermapfname
    x_begin=x_begin; x_end=x_end
    dy=5
    Nleg=10+1
    Ngh=2+1
    
    Nparam=Nleg*Ngh
    
    #def fitfunc(params, polygrid):
    #    p = params.reshape((Nleg,Ngh)).T
    #    x = p*polygrid
    #    return np.sum(x,axis=(2,3))
    
    data, edata, header = read_fits_two(fname)
    nx, ny = data.shape
    coeffcen = np.loadtxt(tracefname)
    coeffstd = np.loadtxt(tracestdfname)
    fibmap = np.loadtxt(fibermapfname)
    npeak, norder = len(coeffcen), fibmap.shape[1]
    # Generate valid pixels: could do this in a more refined way
    xarrlist = [np.arange(nx) for i in range(norder)]
    xarrlist[0] = np.arange(nx-x_begin)+x_begin
    xarrlist[-1] = np.arange(x_end)
    
    ## SECOND TRACE:
    # For each order, fit spatial profile to the full trace simultaneously
    # Following Kelson ghlb.pdf
    # First also fit a function for the stdv
    # ys = (y-ycen)/(ystd)
    # For each fiber:
    # P(ys,xl) = sum(f,g; L_f(xl) H_g(ys) e^(-ys^2/2))
    # F = 1 or 2 (legendre in xl), H = 10(hermite polynomial in ys)
    
    # Go 1 pixel further on each side, ensure an odd number
    Nextract = int(dy)+4 + int(int(dy) % 2 == 0)
    dextract = int(Nextract/2.)
    vround = np.vectorize(lambda x: int(round(x)))
    start = time.time()
    fitparams = np.zeros((npeak,Nleg*Ngh))
    alltmparr = np.full((npeak,nx,Nextract), np.nan)
    alltmperrarr = np.full((npeak,nx,Nextract), np.nan)
    allfitarr = np.full((npeak,nx,Nextract), np.nan)
    allpolygrid = np.zeros((npeak,Nleg,Ngh,nx,Nextract))
    for ipeak in range(npeak):
        # Initialize peak locations, widths, and data arrays
        xarr = xarrlist[ipeak % norder]
        Nxarr = len(xarr)
        Nxy = Nxarr*Nextract
        
        ypeak = np.polyval(coeffcen[ipeak],xarr)
        ystdv = np.polyval(coeffstd[ipeak],xarr)
        assert np.all(ystdv > 0)
        assert np.all(ystdv > 0)
        intypeak = vround(ypeak)
        intymin = intypeak - dextract
        intymax = intypeak + dextract + 1
        yarr = np.array([np.arange(intymin[j],intymax[j]) for j in range(Nxarr)])
        
        # Grab the relevant data into a flat array for fitting
        tmpdat = np.zeros((Nxarr,Nextract))
        tmperr = np.zeros((Nxarr,Nextract))
        for j,jx in enumerate(xarr):
            tmpdat[j,:] =  data[jx,intymin[j]:intymax[j]]
            tmperr[j,:] = edata[jx,intymin[j]:intymax[j]]
        assert np.all(tmperr > 0), (tmperr <= 0).sum()
        alltmparr[ipeak,xarr,:] = tmpdat
        alltmperrarr[ipeak,xarr,:] = tmperr
        # Apply simple estimate of spectrum flux: sum everything
        # and apply this to polygrid
        specestimate = np.sum(tmpdat, axis=1)
        
        # Set up the polynomial grid in xl and ys
        ys = (yarr.T-ypeak)/ystdv
        expys = np.exp(-ys**2/2.)
        xl = (xarr-np.mean(xarr))/(Nxarr/2.)
        polygrid = np.zeros((Nleg,Ngh,Nxarr,Nextract))
        La = [special.eval_legendre(a,xl)*specestimate for a in range(Nleg)]
        GHb = [special.eval_hermitenorm(b,ys)*expys for b in range(Ngh)]
        for a in range(Nleg):
            for b in range(Ngh):
                polygrid[a,b,:,:] = (GHb[b]*La[a]).T
        allpolygrid[ipeak][:,:,xarr,:] = polygrid
        polygrid = polygrid.T
        
        #params0 = np.ones(Nleg*Ngh)
        #params0 = np.zeros((Nleg,Ngh))
        #params0[0,:] = 1
        #params0[:,0] = 1

        # minimize chi2 = ((tmpdat - f(params,xl,ys))/tmperr)**2
        #def minfunc(params):
        #    # sum over a and b, subtract from tmpdat, chi2
        #    funcarr = fitfunc(params,polygrid).T
        #    return np.sum(((tmpdat - funcarr)/tmperr)**2)
        #result = optimize.minimize(minfunc, params0)
        #allfitarr[ipeak][xarr,:] = fitfunc(result.x, polygrid).T
        #fitparams[ipeak] = result.x
        
        # this is a *linear* least squares fit!
        Xarr = np.reshape(polygrid,(Nxy,Nparam))
        yarr = np.ravel(tmpdat)
        # use errors as weights
        # Note that we don't square it b'c wX is squared
        warr = np.ravel(tmperr)**-1
        wXarr = (Xarr.T*warr).T
        wyarr = warr*yarr
        pcov, residues, rank, svals = linalg.lstsq(wXarr, wyarr)
        fitparams[ipeak] = pcov
        fity = pcov.dot(Xarr.T).reshape(Nxarr,Nextract)
        allfitarr[ipeak][xarr,:] = fity
        
        #print("  {}: {:.1f}s".format(ipeak, time.time()-start))
    print("Total took {:.1f}s".format(time.time()-start))



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
    
