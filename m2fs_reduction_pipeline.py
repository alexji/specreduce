from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
import glob, os, sys, time

from m2fs_utils import mrdfits, read_fits_two, write_fits_two, m2fs_load_files_two
from m2fs_utils import gaussfit
from m2fs_utils import m2fs_4amp

def m2fs_make_master_dark(filenames, outfname, exptime=3600., corners_only=False):
    """
    Make a master dark by taking the median of all dark frames
    """
    # Load data
    master_dark, master_darkerr, headers = m2fs_load_files_two(filenames)
    h = headers[0]
    # Rescale to common exptime
    for k in range(len(filenames)):
        dh = headers[k]
        if dh["EXPTIME"] != exptime:
            master_dark[k] = master_dark[k] * exptime/dh["EXPTIME"]
            master_darkerr[k] = master_darkerr[k] * np.sqrt(dh["EXPTIME"]/exptime)
    # Take median
    master_dark = np.median(master_dark, axis=0)
    # Calculate error
    master_darkerr = np.sqrt(np.sum(master_darkerr**2, axis=0))
    
    # HACK: only subtract dark current from the corners
    if corners_only:
        Npixcorner = 200
        mask = np.ones_like(master_dark, dtype=bool)
        Nx, Ny = mask.shape
        mask[0:Npixcorner,0:Npixcorner] = False
        mask[0:Npixcorner,(Ny-Npixcorner):Ny] = False
        mask[(Nx-Npixcorner):Nx,0:Npixcorner] = False
        mask[(Nx-Npixcorner):Nx,(Ny-Npixcorner):Ny] = False
        master_dark[mask] = 0.0
        
    _ = h.pop("EXPTIME")
    h["EXPTIME"] = exptime

    write_fits_two(outfname, master_dark, master_darkerr, h)
    print("Created dark frame with texp={} and wrote to {}".format(
            exptime, outfname))
    if corners_only:
        print("Only dark subtracting in the corners!")

def m2fs_subtract_one_dark(infile, outfile, dark, darkerr, darkheader):
    """ Dark subtraction """
    img, imgerr, header = read_fits_two(infile)
    # Subtract dark
    exptimeratio = header["EXPTIME"]/darkheader["EXPTIME"]
    darksub = img - dark * exptimeratio
    # Adjust the errors
    darksuberr = np.sqrt(imgerr**2 + darkerr**2)
    # Zero negative values: I don't want to do this
    #darksub[darksub < 0] = 0.
    write_fits_two(outfile, darksub, darksuberr, header)
    
def m2fs_make_master_flat(filenames, outfname):
    master_flat, master_flaterr, headers = m2fs_load_files_two(filenames)
    master_flat = np.median(master_flat, axis=0)
    master_flaterr = np.median(master_flaterr, axis=0)
    write_fits_two(outfname, master_flat, master_flaterr, headers[0])
    print("Created master flat and wrote to {}".format(outfname))
    
def m2fs_remove_cosmics(filenames,suffix="crr",output_masks=False):
    """
    Remove cosmic rays by finding huge outliers.
    Replace with the median of all filenames.
    """
    imgain = 1.0 # gain has already been applied in m2fs_biastrim
    sciimgs, scierrs, headers = m2fs_load_files_two(filenames)
    readnoises = [h["ENOISE"] for h in headers]
    exptimes = [h["EXPTIME"] for h in headers]
    # Rescale by exptime
    imgmeds = sciimgs.copy()
    for k in range(len(filenames)):
        exptime = exptimes[k]
        imgmeds[k] = imgmeds[k] / exptime
    bigmedian = np.median(imgmeds, axis=0)
    def make_out_name(fname):
        name = os.path.basename(fname)
        twd = os.path.dirname(fname)
        assert name.endswith(".fits")
        return os.path.join(twd,name[:-5]+suffix+".fits")
    def make_mask_name(fname):
        name = os.path.basename(fname)
        twd = os.path.dirname(fname)
        assert name.endswith(".fits")
        return os.path.join(twd,name[:-5]+"_mask.fits")
    for k in range(len(filenames)):
        # Calculate sigma value = sqrt(readnoise^2+median*gain)
        bigmedian_k = bigmedian * exptimes[k]
        #sigma_d = scierrs[k]
        sigma_d = np.sqrt(readnoises[k]**2 + imgain * bigmedian_k)
        # Pixels > 5 sigma from the median are considered cosmic rays
        # Replace them with the median value
        # They should have a big error already because large counts, so do not change the error?
        mask = sciimgs[k] > 5*sigma_d + bigmedian_k
        sciimgs[k][mask] = bigmedian_k[mask]
        # Write out cleaned data
        headers[k].add_history("m2fs_crbye: replaced {} cosmic ray pixels".format(mask.sum()))
        write_fits_two(make_out_name(filenames[k]), sciimgs[k], scierrs[k], headers[k])
        print("m2fs_crbye: removed {}/{} pixels from {}".format(mask.sum(),mask.size, filenames[k]))
        
        if output_masks:
            hdu = fits.PrimaryHDU(mask.T.astype(int), headers[k])
            hdu.writeto(make_mask_name(fname))
    
def m2fs_trace_orders(fname, expected_fibers, 
                      nthresh=2.0, ystart=0, dx=20, dy=5, nstep=10, degree=4, ythresh=700,
                      make_plot=True):
    data, edata, header = read_fits_two(fname)
    nx, ny = data.shape
    midx = round(nx/2.)
    thresh = nthresh*np.median(data)
    
    # Find peaks at center of CCD
    auxdata = np.zeros(ny)
    for i in range(ny):
        ix1 = int(np.floor(midx-dx/2.))
        ix2 = int(np.ceil(midx+dx/2.))+1
        auxdata[i] = np.median(data[ix1:ix2,i])
    dauxdata = np.gradient(auxdata)
    yarr = np.arange(ny)
    peak1 = np.zeros(ny, dtype=bool)
    for i in range(ny-1):
        if (dauxdata[i] >= 0) and (dauxdata[i+1] < 0) and (auxdata[i] >= thresh) and (i > ystart):
            peak1[i] = True
    peak1 = np.where(peak1)[0]
    npeak = len(peak1)
    peak = np.zeros(npeak)
    for i in range(npeak):
        ix1 = int(np.floor(peak1[i]-dy/2.))
        ix2 = int(np.ceil(peak1[i]+dy/2.))+1
        auxx = yarr[ix1:ix2]
        auxy = auxdata[ix1:ix2]
        coef = gaussfit(auxx, auxy, [auxdata[peak1[i]], peak1[i], 2, thresh/2.])
        peak[i] = coef[1]
    assert npeak==expected_fibers
    # TODO allow some interfacing of the parameters and plotting
    # Trace peaks across dispersion direction
    ypeak = np.zeros((nx,npeak))
    nopeak = np.zeros((nx,npeak))
    ypeak[midx,:] = peak
    start = time.time()
    for i in range(npeak):
        sys.stdout.write("\r")
        sys.stdout.write("TRACING FIBER {} of {}".format(i+1,npeak))
        # Trace right
        for j in range(midx+nstep, nx, nstep):
            ix1 = int(np.floor(j-dx/2.))
            ix2 = int(np.ceil(j+dx/2.))+1
            auxdata0 = np.median(data[ix1:ix2,:], axis=0)
            #auxdata0 = np.zeros(ny)
            #for k in range(ny):
            #    auxdata0[i] = np.median(data[ix1:ix2,k])
            auxthresh = 2*np.median(auxdata0)
            ix1 = max(0,int(np.floor(ypeak[j-nstep,i]-dy/2.)))
            ix2 = min(ny,int(np.ceil(ypeak[j-nstep,i]+dy/2.))+1)
            auxx = yarr[ix1:ix2]
            auxy = auxdata0[ix1:ix2]
            # stop tracing orders that run out of signal
            if (data[j,int(ypeak[j-nstep,i])] <= data[j-nstep,int(ypeak[j-2*nstep,i])]) and \
               (data[j,int(ypeak[j-nstep,i])] <= ythresh):
                break
            if np.max(auxy) >= auxthresh:
                coef = gaussfit(auxx, auxy, [auxdata0[int(ypeak[j-nstep,i])], ypeak[j-nstep,i], 2, thresh/2.],
                                xtol=1e-6,maxfev=10000)
                ypeak[j,i] = coef[1]
            else:
                ypeak[j,i] = ypeak[j-nstep,i] # so i don't get lost
                nopeak[j,i] = 1
        # Trace left
        for j in range(midx-nstep, int(np.ceil(dx/2.))+1, -1*nstep):
            #auxdata0 = np.zeros(ny)
            ix1 = int(np.floor(j-dx/2.))
            ix2 = min(nx, int(np.ceil(j+dx/2.))+1)
            auxdata0 = np.median(data[ix1:ix2,:], axis=0)
            #for k in range(ny):
            #    auxdata0[i] = np.median(data[ix1:ix2,k])
            auxthresh = 2*np.median(auxdata0)
            ix1 = int(np.floor(ypeak[j+nstep,i]-dy/2.))
            ix2 = min(ny, int(np.ceil(ypeak[j+nstep,i]+dy/2.))+1)
            auxx = yarr[ix1:ix2]
            auxy = auxdata0[ix1:ix2]
            # stop tracing orders that run out of signal
            if (data[j,int(ypeak[j+nstep,i])] <= data[j+nstep,int(ypeak[j+2*nstep,i])]) and \
               (data[j,int(ypeak[j+nstep,i])] <= ythresh):
                break
            if np.max(auxy) >= auxthresh:
                coef = gaussfit(auxx, auxy, [auxdata0[int(ypeak[j+nstep,i])], ypeak[j+nstep,i], 2, thresh/2.],
                                xtol=1e-6,maxfev=10000)
                ypeak[j,i] = coef[1]
            else:
                ypeak[j,i] = ypeak[j+nstep,i] # so i don't get lost
                nopeak[j,i] = 1
    ypeak[(nopeak == 1) | (ypeak == 0)] = np.nan
    print("\nTracing took {:.1f}s".format(time.time()-start))
    
    coeff = np.zeros((degree+1,npeak))
    for i in range(npeak):
        sel = np.isfinite(ypeak[:,i]) #np.where(ypeak[:,i] != -666)[0]
        xarr_fit = np.arange(nx)[sel]
        auxcoeff = np.polyfit(xarr_fit, ypeak[sel,i], degree)
        coeff[:,i] = auxcoeff
                
    np.savetxt("{}/{}_tracecoeff.txt".format(
            os.path.dirname(fname),os.path.basename(fname)[:-5]), coeff.T)
    with open("{}/{}_trace.xy".format(
            os.path.dirname(fname),os.path.basename(fname)[:-5]), "w") as fp:
        for i in range(0, nx-1, 10):
            for j in range(npeak):
                fp.write("{}\t{}\n".format(i+1,np.polyval(coeff[:,j],i)+1))
    
    if make_plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(20,20))
        ax.imshow(data.T, origin="lower")
        xarr = np.arange(2048)
        for j in range(npeak):
            ax.plot(xarr, np.polyval(coeff[:,j],xarr), color='orange', lw=.5)
        ax.plot(xarr, ypeak, 'r.', ms=1)
        fig.savefig("{}/{}_trace.png".format(
                os.path.dirname(fname), os.path.basename(fname)[:-5]),
                    dpi=300,bbox_inches="tight")
        plt.close(fig)
    
