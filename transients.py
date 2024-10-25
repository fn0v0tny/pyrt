#!/usr/bin/env python3

import os
import sys
import time
from catalog import Catalog


if sys.version_info[0]*1000+sys.version_info[1]<3008:
    print("Error: python3.8 or higher is required (this is python %d.%d.%d)"%(sys.version_info[0], sys.version_info[1], sys.version_info[2]))
    exit(-1)

import math
import subprocess
import astropy
import astropy.io.fits
import astropy.wcs
import astropy.table
from astropy.coordinates import SkyCoord
import numpy as np
import argparse

import scipy.optimize as fit
from sklearn.neighbors import KDTree,BallTree

import zpnfit
import fotfit

def try_grbt0(target): 
    """tries to run a command that gets T0 of a GRB from the stars DB"""
    try:
            some_file = "tmp%d.grb0"%(os.getppid())
            os.system("grbt0 %d > %s"%(target, some_file))
            f = open(some_file, "r")
            t0=np.float64(f.read())
            f.close()
            return t0
    except:
        return 0

def try_tarname(target): 
    """tries to run a command that gets TARGET name from the stars DB"""
    try:
            some_file = "tmp%d.tmp"%(os.getppid())
            os.system("tarname %d > %s"%(target, some_file))
            f = open(some_file, "r")
            name=f.read()
            f.close()
            return name.strip()
    except:
            return "-"

def isnumber(a):
    try:
        k=int(a)
        return True
    except:
        return False

def readOptions(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="Compute photometric calibration for a FITS image.")
    # Transients specific:
    parser.add_argument("-E", "--early", help="Limit transients to t-t0 < 0.5 d", action='store_true')
    parser.add_argument("-f", "--frame", help="Image frame width to be ignored in pixels (default=10)", type=float)
    parser.add_argument("-g", "--regs", action='store_true', help="Save per image regs")
    parser.add_argument("-s", "--siglim", help="Sigma limit for detections to be taken into account.", type=float)
    parser.add_argument("-m", "--min-found", help="Minimum number of occurences to consider candidate valid", type=int, default=4)
    parser.add_argument("-u", "--usno", help="Use USNO catalog.", action='store_true')
    parser.add_argument("-q", "--usnox", help="Use USNO catalog extra.", action='store_true')
    # General options:
    parser.add_argument("-l", "--maglim", help="Do not get any more than this mag from the catalog to compare.", type=float)
    parser.add_argument("-L", "--brightlim", help="Do not get any less than this mag from the catalog to compare.", type=float)
    parser.add_argument("-i", "--idlimit", help="Set a custom idlimit.", type=float)
    parser.add_argument("-c", "--catalog", action='store', help="Use this catalog as a reference.")
    parser.add_argument("-e", "--enlarge", help="Enlarge catalog search region", type=float)
    parser.add_argument("-v", "--verbose", action='store_true', help="Print debugging info.")
    parser.add_argument("files", help="Frames to process", nargs='+', action='extend', type=str)
    opts = parser.parse_args(args)
    return opts

def simple_color_model(line, data):
    """Add transformed magnitudes based on color model"""
    mag,color1,color2,color3,color4=data
    model=0
    for chunk in line.split(","):
        term,strvalue = chunk.split("=")
        value=np.float64(strvalue)
        if term[0] == 'P':
            pterm = value; n=1;
            for a in term[1:]:
                if isnumber(a): n = int(a)
                if a == 'C': pterm *= np.power(color1, n); n=1;
                if a == 'D': pterm *= np.power(color2, n); n=1;
                if a == 'E': pterm *= np.power(color3, n); n=1;
                if a == 'F': pterm *= np.power(color4, n); n=1;
                if a == 'X' or a == 'Y' or a == 'R': pterm = 0;
            model += pterm 
        if term == 'XC': 
            if value < 0: bval = value * color1; 
            if value > 0 and value <= 1: bval = value * color2; 
            if value > 1: bval = (value-1) * color3 + color2; 
            model += bval;
    return mag+model

def open_ecsv_file(arg, verbose=True):
    """Opens a file if possible, given .ecsv or .fits"""
    det = None
    
    fn = os.path.splitext(arg)[0] + ".ecsv"
    
    try:
        det = astropy.table.Table.read(fn, format="ascii.ecsv")
        det.meta['filename'] = fn;
        return det
    except:
        if verbose: print("%s did not open as an ecsv table"%(fn)); 
        det = None

    return det


def remove_junk(hdr):
    for delme in ['comments','COMMENTS','history','HISTORY']:
        try:
            del hdr[delme]
        except KeyError:
            None
    return


def process_single_image(arg, options, frame, siglim):
    """Process a single image for transient detection"""
    # Open detection file
    det = open_ecsv_file(arg, verbose=options.verbose)
    if det is None:
        if options.verbose:
            print(f"Cannot handle {arg}. Skipping.")
        return None
        
    if options.verbose:
        print("Input file:", det.meta['filename'])
    
    remove_junk(det.meta)
    
    # Initialize WCS and compute sky coordinates
    try:
        imgwcs = astropy.wcs.WCS(det.meta)
        det['ALPHA_J2000'], det['DELTA_J2000'] = imgwcs.all_world2pix(
            [det['X_IMAGE']], [det['Y_IMAGE']], 1)
    except:
        return None

    # Set field size
    try:
        field = det.meta['FIELD']
    except:
        det.meta['FIELD'] = 180
        
    # Early data check
    if options.early:
        t0 = try_grbt0(det.meta['TARGET'])
        if det.meta['CTIME'] + det.meta['EXPTIME']/2 - t0 > 43200:
            return None
            
    # Calculate epoch for proper motion correction 
    epoch = (det.meta['JD'] - 2457204.5) / 365.2425
    
    # Get reference catalogs
    candidates = get_transient_candidates(det, imgwcs, options, epoch, frame, siglim)
    
    if options.regs:
        save_region_file(candidates, det, options)
        
    return candidates

def get_transient_candidates(det, imgwcs, options, epoch, frame, siglim):
    """Get transient candidates by comparing with reference catalogs"""
    # Initialize catalog parameters
    cat_params = {
        'ra': det.meta['CTRRA'],
        'dec': det.meta['CTRDEC'],
        'width': options.enlarge * det.meta['FIELD'] if options.enlarge else det.meta['FIELD'],
        'height': options.enlarge * det.meta['FIELD'] if options.enlarge else det.meta['FIELD'],
        'mlim': options.maglim
    }
    
    # Get ATLAS catalog
    cat = Catalog(catalog='atlas@   ', **cat_params)
    if cat is None or len(cat) == 0:
        return None
        
    # Apply proper motion correction
    cat['radeg'] += epoch * cat['pmra'] 
    cat['decdeg'] += epoch * cat['pmdec']
    
    # Match detections with catalog
    candidates = match_and_filter_detections(det, cat, imgwcs, options, frame, siglim)
    
    # If USNO option enabled, filter candidates against USNO
    if options.usno and len(candidates) > 0:
        candidates = filter_usno_matches(candidates, det, imgwcs, options)
        
    return candidates

def match_and_filter_detections(det, cat, imgwcs, options, frame, siglim):
    """Match detections with catalog and filter for transient candidates"""
    # Transform catalog coordinates to pixel space
    try:
        cat_x, cat_y = imgwcs.all_world2pix(cat['radeg'], cat['decdeg'], 1)
        cat_pixels = np.array([cat_x, cat_y]).transpose()
    except:
        if options.verbose:
            print("Astrometry transform failed")
        return None
        
    # Set up KD-tree for matching
    idlimit = options.idlimit if options.idlimit else det.meta.get('FWHM', 1.2)
    tree = KDTree(cat_pixels)
    
    # Match detections
    det_pixels = np.array([det['X_IMAGE'], det['Y_IMAGE']]).transpose()
    matches_idx, matches_dist = tree.query_radius(det_pixels, 
                                                r=idlimit,
                                                return_distance=True)
    
    # Filter and collect candidates
    candidates = []
    for i, (matches, detection) in enumerate(zip(matches_idx, det)):
        if is_candidate(matches, detection, cat, det.meta, frame, siglim):
            candidates.append(detection)
            
    return astropy.table.Table(candidates)
def is_candidate(matches, detection, catalog, metadata, frame, siglim):
    """Determine if a detection is a transient candidate"""
    # Reject detections near frame edges
    if (detection['X_IMAGE'] < frame or 
        detection['Y_IMAGE'] < frame or
        detection['X_IMAGE'] > metadata['IMGAXIS1'] - frame or
        detection['Y_IMAGE'] > metadata['IMGAXIS2'] - frame):
        return False
        
    # Check photometric quality
    if detection['MAGERR_CALIB'] >= 1.091/siglim:
        return False
        
    # No catalog matches - potential candidate
    if len(matches) == 0:
        return True
        
    # Compare magnitudes with catalog
    for match_idx in matches:
        try:
            # Get colors for transformation
            required_bands = ['Sloan_g', 'Sloan_r', 'Sloan_i', 'Sloan_z', 'J']
            mags = []
            for band in required_bands:
                if band not in catalog.columns:
                    continue
                mag = catalog[match_idx][band]
                if np.isnan(mag) or mag > 99:  # Check for invalid magnitudes
                    continue
                mags.append(mag)
                
            if len(mags) < 2:  # Need at least two magnitudes for color
                continue
                
            # Fill missing magnitudes with estimates if needed
            while len(mags) < 5:
                mags.append(mags[-1])  # Duplicate last magnitude
                
            # Transform catalog magnitude using color model
            cat_mag = simple_color_model(
                metadata['RESPONSE'],
                (mags[1],      # Base magnitude (usually r-band)
                 mags[0] - mags[1],  # g-r color
                 mags[1] - mags[2],  # r-i color
                 mags[2] - mags[3],  # i-z color
                 mags[3] - mags[4])  # z-J color
            )
            
            det_mag = detection['MAG_CALIB']
            
            # If magnitude difference within tolerance, not a candidate
            if abs(cat_mag - det_mag) <= siglim * detection['MAGERR_CALIB']:
                return False
                
        except Exception as e:
            continue  # Skip problematic catalog matches
            
    return True

def filter_usno_matches(candidates, det, imgwcs, options):
    """Filter candidates by checking USNO catalog"""
    usno_params = {
        'ra': det.meta['CTRRA'],
        'dec': det.meta['CTRDEC'],
        'width': options.enlarge * det.meta['FIELD'] if options.enlarge else det.meta['FIELD'],
        'height': options.enlarge * det.meta['FIELD'] if options.enlarge else det.meta['FIELD'],
        'mlim': options.maglim
    }
    
    # Get USNO catalog
    usno = Catalog(catalog='usno', **usno_params)
    if usno is None:
        return candidates
        
    # Set up matching parameters
    phases = [
        ('simple', options.maglim, options.idlimit if options.idlimit else det.meta.get('FWHM', 1.2)),
        ('double', options.maglim - 1, 4),
        ('bright', options.maglim - 9, 10)
    ]
    
    filtered_candidates = candidates
    for phase, maglim, match_radius in phases:
        if len(filtered_candidates) < 1:
            break
            
        # Filter bright USNO stars
        bright_usno = usno[usno['R1mag'] < maglim]
        if len(bright_usno) == 0:
            continue
            
        # Transform USNO coordinates
        usno_x, usno_y = imgwcs.all_world2pix(bright_usno['radeg'], 
                                             bright_usno['decdeg'], 1)
        usno_pixels = np.array([usno_x, usno_y]).transpose()
        
        # Match candidates against USNO
        tree = KDTree(usno_pixels)
        cand_pixels = np.array([filtered_candidates['X_IMAGE'], 
                              filtered_candidates['Y_IMAGE']]).transpose()
        matches = tree.query_radius(cand_pixels, r=match_radius, 
                                  return_distance=True)[0]
                                  
        # Keep only unmatched candidates
        filtered_candidates = filtered_candidates[[len(m) == 0 for m in matches]]
        
        if options.verbose:
            print(f'USNO {phase} left {len(filtered_candidates)} candidates')
            
    return filtered_candidates

def save_region_file(candidates, det, options):
    """Save DS9 region file for detected transients"""
    regfile = os.path.splitext(det.meta['filename'])[0] + "-tr.reg"
    with open(regfile, "w+") as f:
        f.write("# Region file format: DS9 version 4.1\n")
        f.write("global color=green dashlist=8 3 width=3 " +
                "font=\"helvetica 10 normal roman\" select=1 highlite=1 " +
                "dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n")
        
        for t in candidates:
            f.write(f"circle({t['X_IMAGE']:.7f},{t['Y_IMAGE']:.7f}," +
                    f"{5*det.meta.get('FWHM',1.2)*det.meta['PIXEL']}\") # color=red\n")

def movement_residuals(fitvalues, data):
    """Calculate residuals for movement fitting"""
    a0,da=fitvalues 
    t,pos = data
    return a0 + da*t - pos

def main():
    """Main function to process transient detection"""
    options = readOptions(sys.argv[1:])

    if options.maglim is None:
        options.maglim = 20

    if options.verbose:
        print("%s running in python %d.%d.%d" % (
            os.path.basename(sys.argv[0]), 
            sys.version_info[0], 
            sys.version_info[1], 
            sys.version_info[2]
        ))
        print("Magnitude limit set to %.2f" % (options.maglim))

    # Initialize tracking variables
    maxtime = 0.0
    mintime = 1e99
    imgtimes = []
    old = None  # Will store accumulated candidates
    mags = []   # Will store magnitude histories
    imgno = 0
    
    frame = options.frame if options.frame is not None else 10
    siglim = options.siglim if options.siglim is not None else 5

    # Process each input file
    for arg in options.files:
        print("Processing file", arg)
        
        # Process single image for candidates
        candidates = process_single_image(arg, options, frame, siglim)
        if candidates is None:
            continue
            
        # Update time tracking
        ctime = candidates.meta['CTIME']
        exptime = candidates.meta['EXPTIME']
        if ctime < mintime:
            mintime = ctime
        if ctime + exptime > maxtime:
            maxtime = ctime + exptime
        imgtimes.append(ctime + exptime/2)

        # First image or no previous detections
        if old is None:
            old = candidates
            mags = [astropy.table.Table(row) for row in candidates]
            imgno += 1
            continue

        # Cross-match new candidates with previous detections
        tree = BallTree(np.array([old['ALPHA_J2000']*np.pi/180,
                                old['DELTA_J2000']*np.pi/180]).transpose(),
                       metric='haversine')
                       
        new_coords = np.array([candidates['ALPHA_J2000']*np.pi/180,
                             candidates['DELTA_J2000']*np.pi/180]).transpose()
                             
        matches_idx, matches_dist = tree.query_radius(
            new_coords,
            r=candidates.meta['PIXEL']*options.idlimit/3600.0*np.pi/180,
            return_distance=True
        )

        # Update records based on matches
        for i, (matches, dist) in enumerate(zip(matches_idx, matches_dist)):
            if len(matches) > 0:
                # Known source - update existing record
                old['NUM'][matches[0]] += 1
                mags[matches[0]].add_row(candidates[i])
            else:
                # New source - add to records
                old.add_row(candidates[i])
                mags.append(astropy.table.Table(candidates[i]))
                
        imgno += 1

    # Early exit if no transients found
    if old is None or len(old) < 1:
        print("No transients found (old is None)")
        sys.exit(0)

    # Filter candidates based on minimum detections
    sufficient_detections = []
    for i, mag_history in enumerate(mags):
        if len(mag_history) >= options.min_found:
            sufficient_detections.append(i)
    
    if len(sufficient_detections) < 1:
        print(f"No transients found (none makes it > {options.min_found}×)")
        sys.exit(0)

    # Analyze remaining candidates
    transients = []
    for idx in sufficient_detections:
        candidate = mags[idx]
        
        # Calculate time reference
        t0 = np.min(old['CTIME'])/2 + np.max(old['CTIME'])/2
        dt = np.max(old['CTIME']) - np.min(old['CTIME'])
        
        # Analyze movement in RA
        x = candidate['CTIME'] - t0
        ra_data = [x, candidate['ALPHA_J2000']]
        ra_fit = fit.least_squares(movement_residuals, 
                                 [candidate['ALPHA_J2000'][0], 1.0],
                                 args=[ra_data], 
                                 ftol=1e-14)
        
        # Analyze movement in Dec
        dec_data = [x, candidate['DELTA_J2000']]
        dec_fit = fit.least_squares(movement_residuals,
                                  [candidate['DELTA_J2000'][0], 1.0],
                                  args=[dec_data],
                                  ftol=1e-14)
        
        # Calculate movement statistics
        ra_speed = ra_fit.x[1] * 3600 * 3600  # arcsec/h
        dec_speed = dec_fit.x[1] * 3600 * 3600  # arcsec/h
        ra_scatter = np.median(np.abs(movement_residuals(ra_fit.x, ra_data))) / 0.67
        dec_scatter = np.median(np.abs(movement_residuals(dec_fit.x, dec_data))) / 0.67
        
        # Calculate total movement
        dpos = np.sqrt(dec_speed**2 + 
                      (ra_speed * np.cos(dec_fit.x[0]*np.pi/180.0))**2) * (dt/3600.0)
        
        # Calculate magnitude statistics
        weights = 1.0 / candidate['MAGERR_CALIB']**2
        mag0 = np.sum(candidate['MAG_CALIB'] * weights) / np.sum(weights)
        magvar = np.sqrt(np.average(((candidate['MAG_CALIB']-mag0) / 
                                   candidate['MAGERR_CALIB'])**2))

        # Determine status
        status = "!!!"  # Default good
        if dpos > candidates.meta['PIXEL']:
            status = "*  "  # Moving object
        if magvar < 3:
            status = " - " if status == "!!!" else "*- "  # Not variable
            
        # Store results
        transients.append([
            idx,                    # Original index
            len(candidate),         # Number of detections
            ra_fit.x[0],           # RA position
            dec_fit.x[0],          # Dec position
            ra_speed,              # RA proper motion
            dec_speed,             # Dec proper motion
            dpos,                  # Total movement
            ra_scatter*3600,       # RA scatter
            dec_scatter*3600,      # Dec scatter
            np.sqrt(ra_scatter**2 + dec_scatter**2)*3600,  # Total scatter
            mag0,                  # Mean magnitude
            magvar                 # Magnitude variation
        ])
        
        print(f"{status} num:{len(candidate)} mag:{mag0:.2f} magvar:{magvar:.1f}, "
              f"mean_pos:{ra_fit.x[0]:.5f},{dec_fit.x[0]:.5f}, "
              f"movement: {ra_speed*np.cos(dec_fit.x[0]*np.pi/180.0):.3f},"
              f"{dec_speed:.3f}, sigma: {ra_scatter*3600:.2f},{dec_scatter*3600:.2f}")

    # Create final transients table
    trans = astropy.table.Table(
        rows=transients,
        names=['INDEX','NUM','ALPHA_J2000','DELTA_J2000','ALPHA_MOV','DELTA_MOV',
               'DPOS','ALPHA_SIG','DELTA_SIG','SIGMA','MAG_CALIB','MAG_VAR'],
        dtype=['int64','int64','float64','float64','float32','float32',
               'float32','float32','float32','float32','float32','float32']
    )
    print(trans)
    print("In total", len(old), "positions considered.")

if __name__ == "__main__":
    main()