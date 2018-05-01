import numpy as np
import glob, os, sys

from m2fs_utils import mrdfits, read_fits_two, write_fits_two
from m2fs_utils import m2fs_4amp

def m2fs_make_master_dark(filenames, outfname, exptime=3600., corners_only=False):
    """
    Make a master dark by taking the median of all dark frames
    """
    Ndark = len(filenames)
    dark, h = mrdfits(filenames[0], 0)
    Nx, Ny = dark.shape
    
    master_dark = np.empty((Ndark,Nx,Ny))
    master_darkerr = np.empty((Ndark,Nx,Ny))
    for k,fname in enumerate(filenames):
        d, de, dh = read_fits_two(fname)
        if dh["EXPTIME"] != exptime:
            master_dark[k] = d * exptime/dh["EXPTIME"]
            master_darkerr[k] = de * np.sqrt(dh["EXPTIME"]/exptime)
        else:
            master_dark[k] = d
            master_darkerr[k] = d

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

    write_fits_two(outfname, master_dark.T, master_darkerr.T, h)
    print("Created dark frame with texp={} and wrote to {}".format(
            exptime, outfname))
    if corners_only:
        print("Only dark subtracting in the corners!")

def m2fs_subtract_one_dark(infile, outfile, dark, darkerr, darkheader):
    """ Simple dark subtraction """
    img, imgerr, header = read_fits_two(infile)
    # Subtract dark
    exptimeratio = header["EXPTIME"]/darkheader["EXPTIME"]
    darksub = img - dark * exptimeratio
    # Adjust the errors
    darksuberr = np.sqrt(imgerr**2 + darkerr**2)
    
    # Zero negative values: I don't want to do this
    #darksub[darksub < 0] = 0.
    
    write_fits_two(outfile, darksub, darksuberr, header)
    
