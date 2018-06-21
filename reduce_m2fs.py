"""
File name: reduce_m2fs.py
Author: Alex Ji (based on code by T.T. Hansen and J.D. Simon)
Date Created: 4/24/2018
Date last modified: 6/19/2018
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
    
import numpy as np
import glob, os, sys, time
from importlib import reload
from scipy import optimize, signal, ndimage, special, linalg

from astropy.io import fits, ascii
from astropy.stats import biweight_location, biweight_scale

from m2fs_utils import mrdfits, read_fits_two, write_fits_two, m2fs_load_files_two
from m2fs_utils import gaussfit, jds_poly_reject, m2fs_extract1d
from m2fs_utils import m2fs_4amp
from m2fs_utils import make_multispec, parse_idb, calc_wave

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
        # Start and end of the first and last orders
        x_begin = 775
        x_end = 1300
        # Littrow Ghost
        badcolmin = 1000
        badcolmax = 1300
    else:
        raise ValueError(rb)
    rwd = "../raw"
    twd = "../reduction_files"
    
    mdarkfname = "{}/mdark_{}.fits".format(twd,rb)
    mflatfname = "{}/mflat_{}.fits".format(twd,rb)
    mflatscatfname = "{}/mflat_{}_scat.fits".format(twd,rb)
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
    longarcnums = [1412,1415,1421,1429,1433,1566,1572,1576,1580,1584]
    shortarcnums = [1413,1416,1422,1430,1434,1567,1573,1577,1581,1585]
    
    darks = ["{}/{}{:04}.fits".format(rwd,rb,i) for i in darknums]
    flats = ["{}/{}{:04}d.fits".format(twd,rb,i) for i in flatnums]
    sciences = ["{}/{}{:04}d.fits".format(twd,rb,i) for i in scinums]
    scicrrs = ["{}/{}{:04}dcrr.fits".format(twd,rb,i) for i in scinums]
    scicrrss = ["{}/{}{:04}dcrr_scat.fits".format(twd,rb,i) for i in scinums]
    #scicrrfs = ["{}/{}{:04}dcrrf.fits".format(twd,rb,i) for i in scinums]
    long_arcs = ["{}/{}{:04}d.fits".format(twd,rb,i) for i in longarcnums]
    short_arcs = ["{}/{}{:04}d.fits".format(twd,rb,i) for i in shortarcnums]
    
    n1_flats = ["{}/{}{:04}d.fits".format(twd,rb,i) for i in n1_flatnum]
    n2_flats = ["{}/{}{:04}d.fits".format(twd,rb,i) for i in n2_flatnum]
    n1_scis = ["{}/{}{:04}d.fits".format(twd,rb,i) for i in n1_scinum]
    n2_scis = ["{}/{}{:04}d.fits".format(twd,rb,i) for i in n2_scinum]
    n1_scicrrs = ["{}/{}{:04}dcrr.fits".format(twd,rb,i) for i in n1_scinum]
    n2_scicrrs = ["{}/{}{:04}dcrr.fits".format(twd,rb,i) for i in n2_scinum]
    n1_scicrrss = ["{}/{}{:04}dcrr_scat.fits".format(twd,rb,i) for i in n1_scinum]
    n2_scicrrss = ["{}/{}{:04}dcrr_scat.fits".format(twd,rb,i) for i in n2_scinum]
    #n1_scicrrfs = ["{}/{}{:04}dcrrf.fits".format(twd,rb,i) for i in n1_scinum]
    #n2_scicrrfs = ["{}/{}{:04}dcrrf.fits".format(twd,rb,i) for i in n2_scinum]
    
    linelist = ascii.read("bulgegc1_longarc.dat")

#def tmp():
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
    #m2fs_reduction_pipeline.m2fs_trace_orders(n1_mflatfname, expected_fibers)
    #m2fs_reduction_pipeline.m2fs_trace_orders(n2_mflatfname, expected_fibers)
    
    ## Find extraction profile from trace/flat
    ## I am not currently using the GHLB fit profile, instead just the empirical profile
    ## But this does pull out the extraction regions so still has to be run.
    #m2fs_reduction_pipeline.m2fs_ghlb_profile(mflatfname, tracefname, tracestdfname, fibermapfname,
    #                                          x_begin, x_end, make_plot=True)
    
    ## Subtract scattered light
    #m2fs_reduction_pipeline.m2fs_scatlight(mflatfname, mflatfname, fibermapfname, x_begin, x_end, badcolmin, badcolmax)
    #for fname in scicrrs:
    #    m2fs_reduction_pipeline.m2fs_scatlight(fname, mflatfname, fibermapfname, x_begin, x_end, badcolmin, badcolmax)

    ## Rerun GHLB to fit profiles but after subtracting scattered light
    #m2fs_reduction_pipeline.m2fs_ghlb_profile(mflatscatfname, tracefname, tracestdfname, fibermapfname,
    #                                          x_begin, x_end, make_plot=True)
    

def extract(fname, flatfname, fibermapfname, wavecalfname, x_begin, x_end):
#if __name__=="__main__":
    ## Run extraction
    #fname = scicrrss[0]
    #fname = os.path.join(twd, "science_{}_dc.fits".format(rb))
    #flatfname = mflatscatfname
    #wavecalfname = "idb1421d.ms.txt"
    
    outdir = os.path.dirname(fname)
    outname = os.path.basename(fname)[:-5]
    multispec_name = os.path.join(outdir,outname+".ms.fits")    
    
    flatoutdir = os.path.dirname(flatfname)
    flatoutname = os.path.basename(flatfname)[:-5]
    fname_fitparams = os.path.join(flatoutdir,flatoutname+"_ghlb_fitparams.txt")
    fname_allyarr = os.path.join(flatoutdir,flatoutname+"_ghlb_yarr.npy")
    fname_allflatdatarr = os.path.join(outdir,flatoutname+"_ghlb_data.npy")
    fname_allflaterrarr = os.path.join(outdir,flatoutname+"_ghlb_errs.npy")
    allflatdatarr = np.load(fname_allflatdatarr)
    allflaterrarr = np.load(fname_allflaterrarr)
    
    # Load the frame the extract
    data, edata, header = read_fits_two(fname)
    nx, ny = data.shape
    fibmap = np.loadtxt(fibermapfname)
    # Load information about fibers
    npeak, norder = fibmap.shape
    xarrlist = [np.arange(nx) for i in range(norder)]
    xarrlist[0] = np.arange(nx-x_begin)+x_begin
    xarrlist[-1] = np.arange(x_end)
    
    fitparams = np.loadtxt(fname_fitparams) # GHLB fit to profile: not using right now
    wavecalparams = parse_idb(wavecalfname) # Wavelength calibration
    
    # Use trace information to extract relevant 2d arrays into alldatarr and allerrarr
    start = time.time()
    allyarr = np.load(fname_allyarr).astype(int) # trace info
    npeak, Nextract, nx = allyarr.shape
    alldatarr = np.full((npeak,nx,Nextract), np.nan)
    allerrarr = np.full((npeak,nx,Nextract), np.nan)
    # Do a simple sum and horne extraction while we're at it
    allspecsum  = np.full((npeak,nx), np.nan)
    allerrssum  = np.full((npeak,nx), np.nan)
    allspecwsum = np.full((npeak,nx), np.nan)
    allspecwerr = np.full((npeak,nx), np.nan)
    allflatdatsum = np.full((npeak,nx), np.nan)
    allwave     = np.full((npeak,nx), np.nan)
    allspechorne= np.full((npeak,nx), np.nan)
    allprofhorne= np.full((npeak,nx,Nextract), np.nan)
    allerrshorne= np.full((npeak,nx), np.nan)
    allflat     = np.full((npeak,nx), np.nan)
    for ipeak in range(npeak):
        xarr = xarrlist[ipeak % norder]
        yarr = allyarr[ipeak,:,xarr]
        for j, jx in enumerate(xarr):
            alldatarr[ipeak,jx,:] =  data[jx,yarr[j]]
            allerrarr[ipeak,jx,:] = edata[jx,yarr[j]]
        # Wave calibration
        coeffs = wavecalparams[ipeak+1]["coefficients"]
        allwave[ipeak, xarr] = calc_wave(xarr, coeffs)
        # Simple sum
        allspecsum[ipeak,:] = np.sum(alldatarr[ipeak],axis=1)
        allerrssum[ipeak,:] = np.sqrt(np.sum(allerrarr[ipeak]**2,axis=1))
        # Use flat to compute profile for Horne extraction
        # TODO use the GHLB profile here instead?
        flat = allflatdatarr[ipeak]
        allflat[ipeak,:] = np.sum(flat, axis=1) # sum flat profile along extraction aperture
        # Do not normalize to preserve throughput information
        # allflat[ipeak,:] = allflat[ipeak,:]/np.nanpercentile(allflat[ipeak,:],90) 
        flat = (flat.T/np.sum(flat,axis=1)).T # normalize at each wavelength for Horne
        allprofhorne[ipeak] = flat # Hmm this does not account for throughput corrections
        ivar = allerrarr[ipeak]**-2.
        flux = alldatarr[ipeak]
        horneflux = np.sum(flat * flux * ivar, axis=1)/np.sum(flat * flat * ivar, axis=1)
        horneerrs = np.sqrt(np.sum(flat, axis=1)/np.sum(flat * flat * ivar, axis=1))
        allspechorne[ipeak] = horneflux
        allerrshorne[ipeak] = horneerrs
    print("Following trace and simple extraction took {:.1f}s for {}".format(time.time()-start, fname))
    
    allspecsumflat = allspecsum/allflat
    allerrssumflat = allerrssum/allflat
    allspechorneflat = allspechorne/allflat
    allerrshorneflat = allerrshorne/allflat
    
    make_multispec(multispec_name, [allspecsum.T, allerrssum.T, allspecsumflat.T, allerrssumflat.T,
                                    allspechorne.T, allerrshorne.T, allspechorneflat.T, allerrshorneflat.T,
                                    allwave.T, allflat.T], 
                   ["sum flux","sum errs","sum flat","sum flaterr",
                    "horne flux","horne errs","horne flat","horne flaterr",
                    "wave","flat"])
    
#    import matplotlib.pyplot as plt
#    Nrow, Ncol = fibmap.shape
#    fig1, axes1 = plt.subplots(Nrow,Ncol,figsize = (Ncol*6,Nrow*2.5))
#    fig2, axes2 = plt.subplots(Nrow,Ncol,figsize = (Ncol*6,Nrow*2.5))
#    fig3, axes3 = plt.subplots(Nrow,Ncol,figsize = (Ncol*6,Nrow*2.5))
#    fig4, axes4 = plt.subplots(Nrow,Ncol,figsize = (Ncol*6,Nrow*2.5))
#    fig5, axes5 = plt.subplots(Nrow,Ncol,figsize = (Ncol*6,Nrow*2.5))
#    fig6, axes6 = plt.subplots(Nrow,Ncol,figsize = (Ncol*6,Nrow*2.5))
#    vmin1, vmax1 = np.nanpercentile(alldatarr, [1,95])
#    for ipeak, (ax1,ax2,ax3,ax4,ax5,ax6) in \
#            enumerate(zip(axes1.flat, axes2.flat, axes3.flat, 
#                          axes4.flat, axes5.flat, axes6.flat)):
#        z = alldatarr[ipeak]
#        im = ax1.imshow(z.T, origin='lower', aspect='auto', vmin=vmin1, vmax=vmax1)
#        z = alldatarr[ipeak]/allflatdatarr[ipeak]
#        vmin2, vmax2 = np.nanpercentile(z, [.1,90])
#        im = ax2.imshow(z.T, origin='lower', aspect='auto', vmin=vmin2, vmax=vmax2)
#        ## Simple sum extraction
#        ax3.plot(allwave[ipeak],allspecsum[ipeak],color='k',lw=.5)
#        ## Simple weighted mean extraction
#        ax4.plot(allwave[ipeak],allspecwsum[ipeak],color='k',lw=.5)
#        ## Simple weighted mean SNR
#        ax5.plot(allwave[ipeak],allspecwsum[ipeak]/allspecwerr[ipeak],color='k',lw=.5)
#        # Horne extraction
#        ax6.plot(allwave[ipeak],allspechorne[ipeak],color='k',lw=.5)
#    fig1.savefig("test_extract1.png", bbox_inches="tight")
#    fig2.savefig("test_extract2.png", bbox_inches="tight")
#    fig3.savefig("test_extract3.png", bbox_inches="tight")
#    fig4.savefig("test_extract4.png", bbox_inches="tight")
#    fig5.savefig("test_extract5.png", bbox_inches="tight")
#    fig6.savefig("test_extract6.png", bbox_inches="tight")
#    #plt.close(fig1); plt.close(fig2); plt.close(fig3); plt.close(fig4); plt.close(fig5); plt.close(fig6)
#    
#    plt.show()

    ## Apply flat to science frames
    #m2fs_reduction_pipeline.m2fs_flat(scicrrs, mflatfname, tracefname, fibermapfname,
    #                                  x_begin=x_begin, x_end=x_end)
    # TODO why can't I just divide one by the other? Because there's lots of 0s.
    # Without a milky flat, it seems best to just extract the object without applying the flat.
    # Then after also extracting the flat, then divide the 1D spectra.
    # There's no need to do this in 2D!
    ## Coadd
    #m2fs_reduction_pipeline.m2fs_imcombine(scicrrfs, os.path.join(twd, "science_{}_dcf.fits".format(rb)))
    #m2fs_reduction_pipeline.m2fs_imcombine(scicrrs, os.path.join(twd, "science_{}_dc.fits".format(rb)))
    # TODO the better way is to simultaneously fit this out of the individual frames
    #dcfws is the final one
    

if __name__ == "__main__":
    for fname in scicrrss:
        extract(fname, mflatscatfname, fibermapfname, "idb1421d.ms.txt", x_begin, x_end)


def extract_arcs():
    """ Pull out arc multispec (used for IRAF identify) """
    ## extract arcs
    datafname,flatfname,fibermapfname = long_arcs[0], mflatfname, fibermapfname
    data, edata, hdata = read_fits_two(datafname)
    nx,ny = data.shape
    fibmap = np.loadtxt(fibermapfname)
    ntrace = fibmap.size
    nobjs, norder  = fibmap.shape
    
    outdir = os.path.dirname(datafname)
    outname= os.path.basename(datafname)[:-5]
    flatoutdir = os.path.dirname(flatfname)
    flatoutname = os.path.basename(flatfname)[:-5]
    fname_fitparams = os.path.join(flatoutdir,flatoutname+"_ghlb_fitparams.txt")
    fname_allyarr = os.path.join(flatoutdir,flatoutname+"_ghlb_yarr.npy")
    #fname_alldatarr = os.path.join(flatoutdir,flatoutname+"_ghlb_data.npy")
    #fname_allerrarr = os.path.join(flatoutdir,flatoutname+"_ghlb_errs.npy")
    fitparams = np.loadtxt(fname_fitparams)
    allyarr = np.load(fname_allyarr).astype(int)
    Nextract = allyarr.shape[1]
    
    assert fitparams.shape[0] == ntrace
    assert allyarr.shape[0] == ntrace
    
    # Generate valid pixels: could do this in a more refined way
    xarrlist = [np.arange(nx) for i in range(norder)]
    xarrlist[0] = np.arange(nx-x_begin)+x_begin
    xarrlist[-1] = np.arange(x_end)

    # for each trace, pull out yarr from the data
    onedarcs = np.zeros((ntrace,nx))
    for itrace in range(ntrace):
        xarr = xarrlist[itrace % norder]
        yarr = allyarr[itrace][:,xarr]
        darr = np.array([data[xarr,_yarr] for _yarr in yarr])
        # simply collapse for now, though could do something better if needed
        onedarcs[itrace,xarr] = np.sum(darr, axis=0)
    onedarcs = onedarcs - np.nanmedian(onedarcs, axis=0)
    make_multispec(os.path.join(outdir,outname+".ms.fits"), [onedarcs.T], ["lamp spectrum"])
    refobj = 12
    refarcs = onedarcs[(refobj*norder):(refobj+1)*norder,:]
    #refarcs = refarcs - np.nanmedian(refarcs,axis=0)
    
    ## Used IRAF identify to get wavelength solutions
    
    
def tmp():
    # For each spectrum, find the offset that will give the best 
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(nobjs//2, 2, figsize=(16,4*nobjs/2))
    corrarr = np.zeros((ntrace,nx))
    offsets = np.zeros(ntrace)
    for iobj in range(nobjs):
        ax = axes.flat[iobj]
        ax.set_title(str(iobj))
        # this is the slice
        iy1, iy2 = iobj*norder, (iobj+1)*norder
        for iorder in range(norder):
            xarr = xarrlist[iorder]
            y = onedarcs[iy1+iorder,xarr]
            y = y - np.nanmean(y)
            # cross correlate. Doesn't seem to care about saturated lines
            xmid = np.arange(len(xarr))-len(xarr)//2
            corr = signal.correlate(y,refarcs[iorder,xarr],"same")
            l, = ax.plot(xmid,corr)
            corrarr[iobj*norder + iorder, xarr] = corr
            imax = np.argmax(corr)
            out = gaussfit(xmid,corr,[np.max(corr),xmid[imax],10])
            ax.set_ylim(ax.get_ylim())
            ax.plot([out[1],out[1]],ax.get_ylim(),color='k',lw=1)
            offsets[iobj*norder+iorder] = out[1]
            
    fig.savefig("correlation.png",dpi=300)
    plt.close(fig)
    
    # My ThArNe arcs are highly saturated in the red (even for the short exposures).
    # So I need to mask out the saturated Ne lines.
    # If a feature is similar strength across all orders for one object, it is from saturated overspilling
    # Use this to construct a mask manually for the 90s arcs.
    maskranges = [(22,33),(74,90),(232,245),(259,270),(364,377),(425,437),(605,620),
                  (700,712),(770,785),(1048,1065),(1110,1141),(1315,1335),
                  (1373,1400),(1471,1500),(1611,1620),(1629,1654),(1823,1855)]
    mask = np.zeros_like(onedarcs, dtype=bool)
    for mr in maskranges:
        for itrace, offset in enumerate(offsets):
            mr0 = int(mr[0]+offset)
            mr1 = int(mr[1]+offset)
            mask[itrace,mr0:mr1] = True
    maskarcs = onedarcs.copy()
    maskarcs[mask] = np.nan
    fig, ax = plt.subplots()
    plt.imshow(maskarcs,origin='lower',aspect='auto')
    plt.colorbar()
    fig.savefig("maskarcs.png")
    plt.close(fig)
    
    # find peak locations
    all_peak_locations = []
    fig, axes = plt.subplots(nobjs, norder, figsize=(8*norder,3*nobjs))
    for itrace in range(ntrace):
        iorder = itrace % norder
        iobj = int(itrace/norder)
        xarr = xarrlist[iorder]
        yarr = maskarcs[itrace][xarr]
        
        mask = np.isnan(yarr)
        yarr[mask] = 0.
        yarr = yarr - ndimage.filters.median_filter(yarr, 100)
        yarr[mask] = 0.
        
        dyarr = np.gradient(yarr)
        noise = biweight_scale(yarr)
        thresh = 10. * noise
        
        this_linelist = linelist[linelist["order"] == iorder+1]
        w0s = this_linelist["wavetrue"]
        x0s = this_linelist["X"] + offsets[itrace]
        
        ii1 = yarr > thresh
        ii2 = dyarr >= 0
        ii3 = np.zeros_like(ii2)
        ii3[:-1] = dyarr[1:] < 0
        peaklocs = ii1 & ii2 & ii3
        peaklocs[mask] = False
        peakindices = np.where(peaklocs)[0]
        numpeaks = peaklocs.sum()
        print("{:3}: noise={:.3f}, {} peaks".format(itrace, noise, numpeaks))
        peak_locations = []
        window = 5 # pixel window for fitting peak location
        maxpixdiff = 17 # pixel window for matching to a line
        for ipeak, ix in enumerate(peakindices):
            _xx = xarr[ix-window:ix+window+1]
            _yy = yarr[ix-window:ix+window+1]
            xloc = xarr[ix]
            guess = [yarr[ix],xloc,2,0.]
            try:
                popt = gaussfit(_xx,_yy,guess)
            except (RuntimeError, TypeError):
                print("       Failed to fit trace {} line {}/{} at {}".format(itrace, ipeak, numpeaks, xloc))
                continue
            if np.abs(xloc-popt[1]) > 3 or popt[0] < 0:
                print("       Bad fit for trace {} line {}/{} at {}".format(itrace, ipeak, numpeaks, xloc))
                continue
            closest_line = np.argmin(np.abs(x0s - popt[1]))
            x0 = x0s[closest_line]
            w0 = w0s[closest_line]
            if np.abs(x0-popt[1]) > maxpixdiff:
                print("       Bad line match for trace {} line {}/{} at {}: w0={:.1f} x0={:.1f} fit={:.1f}".format(itrace, ipeak, numpeaks, xloc, w0, x0, popt[1]))
                continue
            peak_locations.append((xloc, x0, w0, popt[1], popt[0], popt[2]))
        ax = axes[iobj, iorder]
        ax.plot(xarr, yarr, lw=.7)
        for loc in peak_locations:
            ax.axvline(loc[1],color='r', lw=.3)
            ax.axvline(loc[3],color='b', lw=.3)
        all_peak_locations.append(peak_locations)
        for x0 in x0s:
            ax.axvline(x0, 0,.1, color='k', lw=.3)
        for mr in maskranges:
            mr0 = int(mr[0]+offsets[itrace])
            mr1 = int(mr[1]+offsets[itrace])
            ax.axvspan(mr0,mr1,0,1,color='grey',alpha=.3)
        ax.set_xlim(xarr[0],xarr[-1])

    fig.savefig("arcmatch.pdf", bbox_inches="tight")
    print(list(map(len, all_peak_locations)))

    # find line locations in each trace
    #all_line_locations = []
    #for itrace in range(ntrace):
    #    iorder = itrace % norder
    #    iobj = int(itrace/norder)
    #    xarr = xarrlist[iorder]
    #    yarr = maskarcs[itrace][xarr]
    #    
    #    # Set masked regions to 0, subtract continuum
    #    yarr[np.isnan(yarr)] = 0.
    #    yarr = yarr - ndimage.filters.median_filter(yarr, 100)
    #
    #    this_linelist = linelist[linelist["order"] == iorder+1]
    #    offset = offsets[itrace]
    #    # Find the actual location of each line
    #    line_locations = []
    #    for iline, line in enumerate(this_linelist):
    #        # Initial guess
    #        x0 = offset + line["X"]
    #        w0 = line["wavetrue"]
    #        ibest = np.argmin(np.abs(x0-xarr))
    #        try:
    #            popt = gaussfit(xarr,yarr,[yarr[ibest], x0, 2.])
    #        except RuntimeError:
    #            print("Failed to fit trace {} line {}".format(itrace, iline))
    #            line_locations.append((w0,x0,np.nan))
    #        else:
    #            line_locations.append((w0,x0,popt[1]))
    #    all_line_locations.append(line_locations)
    #
    #    ax = axes[iobj, iorder]
    #    ax.plot(xarr, yarr, lw=1)
    #    for loc in line_locations:
    #        if np.abs(loc[2]-loc[1]) < 5:
    #            ax.axvline(loc[1],color='r', lw=.5)
    #            ax.axvline(loc[2],color='b', lw=.5)
    #fig.savefig("arcmatch.pdf", bbox_inches="tight")
    
    # Get a 0th order wavelength solution as initialization.
    # This is calculated for refobj manually in IRAF.
    #wavelength_solution_iraf = np.load("wavelength_solution_iraf_{}.npy".format(rb))
    
    

    # widths=[
    #signal.find_peaks_cwt(onedarcs[itrace], [widths])
    #fig,axes = plt.subplots(6,figsize=(8,12))
    #for i in range(6):
    #    ax = axes[i]
    #    xarr = xarrlist[i]
    #    ax.plot(xarr, maskarcs[i,xarr], lw=1)
    #    peaks = signal.find_peaks_cwt(maskarcs[i,xarr],range(4,10))
    #    peaks2 = signal.find_peaks_cwt(maskarcs[i,xarr],range(20,30))
    #    if i == 0: peaks += min(xarr)
    #    ax.vlines(peaks-1,0,2e5,color='k',linewidth=.5)
    #    ax.vlines(peaks2-1,0,2e5,color='r',linewidth=.5)
    #    ax.set_ylim(-1e3,1e5)
    #plt.show()
    #def find_peaks(y):
    #    pass
    # default wavelength solution for specific fibers
    
    

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
    
