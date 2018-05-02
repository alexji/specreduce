from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
import glob, os, sys

from m2fs_utils import mrdfits, read_fits_two, write_fits_two, m2fs_load_files_two
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
    
