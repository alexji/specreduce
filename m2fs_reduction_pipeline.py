from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
import glob, os, sys, time

from scipy import optimize, signal, ndimage, special, linalg
from astropy.io import fits
from astropy.stats import biweight

from m2fs_utils import mrdfits, read_fits_two, write_fits_two, m2fs_load_files_two
from m2fs_utils import gaussfit, jds_poly_reject, m2fs_extract1d
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
    # Take median + calculate error
    master_dark = np.median(master_dark, axis=0)
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
    else:
        #err = np.std(master_dark)
        #med = np.median(master_dark)
        ## keep 5% of points
        #mask = master_dark < np.percentile(master_dark,[95])[0]
        ##mask = (master_dark < 0) | (np.abs(master_dark) < med+3*err)
        ##mask = np.abs(master_dark-med) < err
        #print(np.sum(mask),mask.size,np.sum(mask)/mask.size)
        #master_dark[mask] = 0.
        #master_darkerr[mask] = 0.
        pass
        
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
        mask = np.abs(sciimgs[k]) > 5*sigma_d + bigmedian_k
        sciimgs[k][mask] = bigmedian_k[mask]
        # Write out cleaned data
        headers[k].add_history("m2fs_crbye: replaced {} cosmic ray pixels".format(mask.sum()))
        write_fits_two(make_out_name(filenames[k]), sciimgs[k], scierrs[k], headers[k])
        print("m2fs_crbye: removed {}/{} pixels from {}".format(mask.sum(),mask.size, filenames[k]))
        
        if output_masks:
            hdu = fits.PrimaryHDU(mask.T.astype(int), headers[k])
            hdu.writeto(make_mask_name(fname))
    
def m2fs_trace_orders(fname, expected_fibers, 
                      nthresh=2.0, ystart=0, dx=20, dy=5, nstep=10, degree=4, ythresh=500,
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

    ## FIRST TRACE: do in windows
    # Trace peaks across dispersion direction
    ypeak = np.zeros((nx,npeak))
    ystdv = np.zeros((nx,npeak))
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
                ystdv[j,i] = min(coef[2],dy/2.)
            else:
                ypeak[j,i] = ypeak[j-nstep,i] # so i don't get lost
                ystdv[j,i] = ystdv[j-nstep,i]
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
                ystdv[j,i] = min(coef[2], dy/2.)
            else:
                ypeak[j,i] = ypeak[j+nstep,i] # so i don't get lost
                ystdv[j,i] = ystdv[j+nstep,i]
                nopeak[j,i] = 1
    ypeak[(nopeak == 1) | (ypeak == 0)] = np.nan
    ystdv[(nopeak == 1) | (ypeak == 0)] = np.nan
    print("\nTracing took {:.1f}s".format(time.time()-start))
    
    coeff = np.zeros((degree+1,npeak))
    coeff2 = np.zeros((degree+1,npeak))
    for i in range(npeak):
        sel = np.isfinite(ypeak[:,i]) #np.where(ypeak[:,i] != -666)[0]
        xarr_fit = np.arange(nx)[sel]
        auxcoeff = np.polyfit(xarr_fit, ypeak[sel,i], degree)
        coeff[:,i] = auxcoeff
        auxcoeff2 = np.polyfit(xarr_fit, ystdv[sel,i], degree)
        coeff2[:,i] = auxcoeff2

    np.savetxt("{}/{}_tracecoeff.txt".format(
            os.path.dirname(fname),os.path.basename(fname)[:-5]), coeff.T)
    np.savetxt("{}/{}_tracestdcoeff.txt".format(
            os.path.dirname(fname),os.path.basename(fname)[:-5]), coeff2.T)
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
            ax.errorbar(xarr, ypeak[:,j], yerr=ystdv[:,j], fmt='r.', ms=1, elinewidth=.5)
        fig.savefig("{}/{}_trace.png".format(
                os.path.dirname(fname), os.path.basename(fname)[:-5]),
                    dpi=300,bbox_inches="tight")
        plt.close(fig)
        
        fig, ax = plt.subplots(figsize=(8,8))
        stdevs = np.zeros((nx,npeak))
        for j in range(npeak):
            stdevs[:,j] = np.polyval(coeff2[:,j],xarr)
        im = ax.imshow(stdevs.T, origin="lower", aspect='auto')
        fig.colorbar(im)
        fig.savefig("{}/{}_stdevs.png".format(
                os.path.dirname(fname), os.path.basename(fname)[:-5]),
                    dpi=300,bbox_inches="tight")
        plt.close(fig)
    
def m2fs_ghlb_profile(fname, tracefname, tracestdfname, fibermapfname,
                      x_begin, x_end,
                      dy=5, legorder=6, ghorder=10,
                      make_plot=True):
    """ Use the same dy as in m2fs_trace_orders """
    Nleg = legorder + 1
    Ngh = ghorder + 1

    outdir = os.path.dirname(fname)
    outname = os.path.basename(fname)[:-5]
    fname_fitparams = os.path.join(outdir,outname+"_ghlb_fitparams.txt")
    fname_allyarr = os.path.join(outdir,outname+"_ghlb_yarr.npy")
    fname_alldatarr = os.path.join(outdir,outname+"_ghlb_data.npy")
    fname_allerrarr = os.path.join(outdir,outname+"_ghlb_errs.npy")

    Nparam=Nleg*Ngh
    
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
    # But fitting each fiber separately in some specified extraction region
    
    # Go 1 pixel further on each side, ensure an odd number
    Nextract = int(dy)+4 + int(int(dy) % 2 == 0)
    dextract = int(Nextract/2.)
    vround = np.vectorize(lambda x: int(round(x)))
    start = time.time()
    # Outputs
    allyarr = np.zeros((npeak,Nextract,nx))
    fitparams = np.zeros((npeak,Nleg*Ngh))
    alldatarr = np.full((npeak,nx,Nextract), np.nan)
    allerrarr = np.full((npeak,nx,Nextract), np.nan)
    # Plotting saves
    allfitarr = np.full((npeak,nx,Nextract), np.nan)
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
        allyarr[ipeak,:,xarr] = yarr
        
        # Grab the relevant data into a flat array for fitting
        datarr = np.zeros((Nxarr,Nextract))
        errarr = np.zeros((Nxarr,Nextract))
        for j,jx in enumerate(xarr):
            datarr[j,:] =  data[jx,intymin[j]:intymax[j]]
            errarr[j,:] = edata[jx,intymin[j]:intymax[j]]
        assert np.all(errarr > 0), (errarr <= 0).sum()
        alldatarr[ipeak,xarr,:] = datarr
        allerrarr[ipeak,xarr,:] = errarr
        # Apply simple estimate of spectrum flux: sum everything
        specestimate = np.sum(datarr, axis=1)
        
        # Set up the polynomial grid in xl and ys
        ys = (yarr.T-ypeak)/ystdv
        expys = np.exp(-ys**2/2.)
        xl = (xarr-np.mean(xarr))/(Nxarr/2.)
        polygrid = np.zeros((Nxarr,Nextract,Nleg,Ngh))
        La = [special.eval_legendre(a,xl)*specestimate for a in range(Nleg)]
        GHb = [special.eval_hermitenorm(b,ys)*expys for b in range(Ngh)]
        for a in range(Nleg):
            for b in range(Ngh):
                polygrid[:,:,a,b] = (GHb[b]*La[a]).T
        polygrid = np.reshape(polygrid,(Nxy,Nparam))
        
        # this is a *linear* least squares fit!
        Xarr = np.reshape(polygrid,(Nxy,Nparam))
        Yarr = np.ravel(datarr)
        # use errors as weights. note that we don't square it, the squaring happens later
        warr = np.ravel(errarr)**-1
        wXarr = (Xarr.T*warr).T
        wYarr = warr*Yarr
        pfit, residues, rank, svals = linalg.lstsq(wXarr, wYarr)
        fitparams[ipeak] = pfit
        fity = pfit.dot(Xarr.T).reshape(Nxarr,Nextract)
        allfitarr[ipeak][xarr,:] = fity
    print("Total took {:.1f}s".format(time.time()-start))
    np.savetxt(fname_fitparams,fitparams)
    np.save(fname_allyarr,allyarr)
    np.save(fname_alldatarr,alldatarr)
    np.save(fname_allerrarr,allerrarr)

    if make_plot:
        import matplotlib.pyplot as plt
        Nrow, Ncol = fibmap.shape
        fig1, axes1 = plt.subplots(Nrow,Ncol,figsize = (Ncol*6,Nrow*2.5))
        fig2, axes2 = plt.subplots(Nrow,Ncol,figsize = (Ncol*6,Nrow*2.5))
        fig3, axes3 = plt.subplots(Nrow,Ncol,figsize = (Ncol*6,Nrow*2.5))
        fig4, axes4 = plt.subplots(Nrow,Ncol,figsize = (Ncol*6,Nrow*2.5))
        zplot = np.linspace(-6,6)
        normz = np.exp(-0.5*zplot**2)/(2*np.sqrt(np.pi))
        vmax3 = np.nanpercentile(alldatarr,[99.9])[0]
        vmin4 = np.nanpercentile(allerrarr,[10])[0]
        vmax4 = np.nanpercentile(allerrarr,[90])[0]
        for ipeak, (ax1,ax2,ax3,ax4) in enumerate(zip(axes1.flat,axes2.flat,axes3.flat,axes4.flat)):
            z = (alldatarr[ipeak].T-allfitarr[ipeak].T)/allerrarr[ipeak].T
            im = ax1.imshow(z, origin='lower', aspect='auto',vmin=-6,vmax=6)
            ax2.hist(np.ravel(z[np.isfinite(z)]),bins=zplot,normed=True)
            ax2.plot(zplot,normz,lw=.5)
            im = ax3.imshow(alldatarr[ipeak].T, origin='lower', aspect='auto',vmin=0,vmax=vmax3)
            im = ax4.imshow(alldatarr[ipeak].T, origin='lower', aspect='auto',vmin=vmin4,vmax=vmax4)
        outpre = outdir+"/"+outname
        fig1.savefig(outpre+"_ghlbresid_zimg.png",bbox_inches='tight')
        fig2.savefig(outpre+"_ghlbresid_hist.png",bbox_inches='tight')
        fig3.savefig(outpre+"_ghlbresid_img.png",bbox_inches='tight')
        fig4.savefig(outpre+"_ghlbresid_err.png",bbox_inches='tight')
        plt.close(fig1); plt.close(fig2); plt.close(fig3); plt.close(fig4)

def m2fs_flat(scifnames, flatfname, tracefname, fibermapfname,
              yaper = 7, x_begin=900, x_end=1300):
    owd = os.path.dirname(flatfname)
    flatname = os.path.basename(flatfname)[:-5]

    yaper_arr = np.arange(1,1+yaper)
    sciimgs, scierrs, sciheaders = m2fs_load_files_two(scifnames)
    flatimg, flaterr, flatheaders= read_fits_two(flatfname)
    tracecoefs = np.loadtxt(tracefname)
    fibmap = np.loadtxt(fibermapfname)
    norder = fibmap.shape[1]
    
    nfib = len(tracecoefs)
    nx, ny = flatimg.shape
    xarr = np.arange(nx)
    
    ypeak = np.zeros((nx,nfib))
    for i in range(nfib):
        ypeak[:,i] = np.polyval(tracecoefs[i,:], xarr)
    vround = np.vectorize(lambda x: int(round(x)))
    ymins = vround(ypeak) - int(yaper/2.)
    
    # Generate valid pixels
    xarrlist = [np.arange(nx) for i in range(norder)]
    xarrlist[0] = np.arange(nx-x_begin)+x_begin
    xarrlist[-1] = np.arange(x_end)
    
    # Do the fibers from bottom to top
    flatsun = np.ones((nx,ny))
    for i in range(nfib):
        sys.stdout.write("\r  Running fib {} order {}".format(i+1, i % norder + 1))
        xarr = xarrlist[i % norder]
        auxarr = np.zeros(nx)
        for j in xarr:
            ymin = ymins[j,i]
            auxarr[j] = np.max(flatimg[j,ymin:(ymin+yaper-1)])
        auxarr_max = np.max(auxarr)
        auxarr_norm = auxarr/auxarr_max
        # Create flat with each row equal to max in each column
        for k in range(yaper):
            for j in xarr:
                ymin = ymins[j,i]
                flatsun[j,ymin+k] = auxarr_norm[j]
    flatclean = flatimg/flatsun
    print("\nFinished flatsun, flatclean")
    fits.PrimaryHDU(flatsun).writeto(os.path.join(owd,flatname+"_flatsun.fits"), overwrite=True)
    fits.PrimaryHDU(flatclean).writeto(os.path.join(owd,flatname+"_flatclean.fits"), overwrite=True)
    
    # Get another flat
    start = time.time()
    flatsmooth_all = np.zeros((nx,ny))
    flatsmooth_correct = np.zeros((nx,ny))
    flatpix = np.zeros((nx,ny))
    for i in range(nfib):
        sys.stdout.write("\r  Running fib {} order {}".format(i+1, i % norder + 1))
        mean_xarr = np.zeros(nx)
        max_xarr = np.zeros(nx)
        xarr = xarrlist[i % norder]
        for j in xarr:
            ymin = ymins[j,i]
            max_xarr[j] = np.max(flatclean[j,ymin:(ymin+yaper)])
            mean_xarr[j]=np.mean(flatclean[j,ymin:(ymin+yaper)])
        factor = np.mean(max_xarr)/np.mean(mean_xarr)
        corr_xarr = mean_xarr*factor
        
        flatsmooth = np.zeros(nx)
        yfit, coeff = jds_poly_reject(xarr,corr_xarr[xarr], 6, 2, 2)
        smoothfit_coeff=coeff
        flatsmooth[xarr]=np.polyval(smoothfit_coeff, xarr)
        
        # a flat with all rows in each order equal to above fit
        for k in range(yaper):
            for j in xarr:
                ymin = ymins[j,i]
                flatsmooth_all[j,ymin+k] = flatsmooth[j]
        # Create flat with fit in y direction at all places
        #for j in xarr:
            #yflat_arr = np.zeros(yaper)
            #ymin = ymins[j,i]
            #yflat_arr[0:yaper] = flatimg[j,ymin:(ymin+yaper)]
            
            ## normalize y array
            #yflat_arr_max=np.max(yflat_arr)
            #yflat_arr_norm=yflat_arr/yflat_arr_max
            ## poly fit to spectra spatial profile at each x
            #yarr_coeff = np.polyfit(yaper_arr, yflat_arr_norm, 2)
            #yarr_fit=np.polyval(yarr_coeff,yaper_arr)
            #flatsmooth_correct[j,ymin:(ymin+yaper)] = flatsmooth_all[j,(ymin+yaper)]
        for j in xarr:
            ymin = ymins[j,i]
            #flatpix[j,ymin:(ymin+yaper)] = flatclean[j,ymin:(ymin+yaper)]/(flatsmooth_correct[j,ymin:(ymin+yaper)])
            flatpix[j,ymin:(ymin+yaper)] = flatclean[j,ymin:(ymin+yaper)]/(flatsmooth_all[j,ymin:(ymin+yaper)])
    print("\nFinished flatpix ({:.1f}s)".format(time.time()-start))
    fits.PrimaryHDU(flatpix).writeto(os.path.join(owd,flatname+"_flatpix.fits"), overwrite=True)
    flatpix_corr = flatpix
    flatpix_corr[flatpix <= 0] = 1
    
    outs = sciimgs/flatpix_corr
    outerrs = scierrs/flatpix_corr
    for k, scifname in enumerate(scifnames):
        outdir = os.path.dirname(scifname)
        outname= os.path.basename(scifname)[:-5]
        outfname = outdir+"/"+outname+"f.fits"
        print("Writing",outfname)
        sciheaders[k].add_history("m2fs_flat: Applied flat field")
        sciheaders[k].add_history("m2fs_flat: Flat: {}".format(flatfname))
        write_fits_two(outfname, outs[k], outerrs[k], sciheaders[k])
                

def m2fs_imcombine(scifnames, outfname):
    sciimgs, scierrs, sciheaders = m2fs_load_files_two(scifnames)
    w = scierrs**-2.
    wtot = np.sum(w, axis=0)
    scisum = np.sum((sciimgs*w),axis=0)/wtot
    scierr = np.sqrt(1/wtot)
    # TODO fix header!!!
    print("Combined science frames to {}".format(outfname))
    print("TODO: fix header!!!")
    write_fits_two(outfname, scisum, scierr, sciheaders[0])
