#!/usr/bin/env python3

import argparse
import concurrent.futures
import logging
import os
import subprocess
import sys
import time
<<<<<<< HEAD
import logging
import argparse
from contextlib import suppress
import subprocess

import numpy as np

import astropy
import astropy.io.fits
import astropy.wcs
from astropy.table import Table
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, AltAz, EarthLocation

import concurrent.futures
from copy import deepcopy
=======
>>>>>>> cf38f76 (Add the daemon to transient analyser)

# This is to silence a particular annoying warning (MJD not present in a fits file)
import warnings
from contextlib import suppress
from copy import deepcopy

import astropy
import astropy.io.fits
import astropy.table
import astropy.units as u
import astropy.wcs
import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from astropy.wcs import FITSFixedWarning

warnings.simplefilter("ignore", category=FITSFixedWarning)

<<<<<<< HEAD
import fotfit
from catalog import Catalog
=======
from sklearn.neighbors import KDTree

import fotfit
import zpnfit
>>>>>>> cf38f76 (Add the daemon to transient analyser)
from cat2det import remove_junk

# from catalogs import get_atlas, get_catalog
from catalog import Catalog
from config import parse_arguments
<<<<<<< HEAD
from data_handling import PhotometryData, make_pairs_to_fit, compute_zeropoints_all_filters
from match_stars import process_image_with_dynamic_limits
from stepwise_regression import perform_stepwise_regression, parse_terms, expand_pseudo_term

from plotting import create_residual_plots, create_correction_volume_plots
from filter_matching import determine_filter
=======

# from config import load_config
from data_handling import PhotometryData, compute_initial_zeropoints, make_pairs_to_fit
from file_utils import try_det, try_img, write_region_file
from refit_astrometry import refit_astrometry
>>>>>>> cf38f76 (Add the daemon to transient analyser)

if sys.version_info[0] * 1000 + sys.version_info[1] < 3008:
    print(
        f"Error: python3.8 or higher is required (this is python {sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]})"
    )
    sys.exit(-1)


def airmass(z):
    """ Compute astronomical airmass according to Rozenberg(1966) """
    cz = np.cos(z)
    return 1 / (cz + 0.025 * np.exp(-11 * cz))


def summag(magarray):
    """add two magnitudes of the same zeropoint"""
    return -2.5 * np.log10(np.sum(np.power(10.0, -0.4 * np.array(magarray))))


def print_image_line(det, flt, Zo, Zoe, target=None, idnum=0):
    """ print photometric status for an image """
    if target == None:
        tarmag = Zo + det.meta["LIMFLX3"]
        tarerr = ">"
        tarstatus = "not_found"
    else:
        tarmag = Zo + target["MAG_AUTO"]
        tarerr = "%6.3f" % (target["MAGERR_AUTO"])
        tarstatus = "ok"

<<<<<<< HEAD
    print("%s %14.6f %14.6f %s %3.0f %6.3f %4d %7.3f %6.3f %7.3f %6.3f %7.3f %s %d %s %s"%(
        det.meta['FITSFILE'], det.meta['JD'], det.meta['JD']+det.meta['EXPTIME']/86400.0/2, flt, det.meta['EXPTIME'],
        det.meta['AIRMASS'], idnum, Zo, Zoe, (det.meta['LIMFLX10']+Zo),
        (det.meta['LIMFLX3']+Zo), tarmag, tarerr, 0, det.meta['OBSID'], tarstatus))

    return

=======
    print(
        "%s %14.6f %14.6f %s %3.0f %6.3f %4d %7.3f %6.3f %7.3f %6.3f %7.3f %s %d %s %s"
        % (
            det.meta["FITSFILE"],
            det.meta["JD"],
            det.meta["JD"] + det.meta["EXPTIME"] / 86400.0,
            flt,
            det.meta["EXPTIME"],
            det.meta["AIRMASS"],
            idnum,
            Zo,
            Zoe,
            (det.meta["LIMFLX10"] + Zo),
            (det.meta["LIMFLX3"] + Zo),
            tarmag,
            tarerr,
            0,
            det.meta["OBSID"],
            tarstatus,
        )
    )

    return


def get_base_filter(det, options, catalog_info=None):
    """
    Set up or guess the base filter to use for fitting.
    
    Args:
        det: Detection table with metadata
        options: Command line options
        catalog_info: Optional catalog information from Catalog
    
    Returns:
        tuple: (base_filter_name, photometric_system)
    """
    # Define known filter mappings
    FILTER_MAPPINGS = {
        # Standard names
        "Sloan_g": "Sloan_g",
        "Sloan_r": "Sloan_r",
        "Sloan_i": "Sloan_i",
        "Sloan_z": "Sloan_z",
        "Johnson_B": "Johnson_B",
        "Johnson_V": "Johnson_V",
        "Johnson_R": "Johnson_R",
        "Johnson_I": "Johnson_I",
        # Alternative names
        "g-SLOAN": "Sloan_g",
        "r-SLOAN": "Sloan_r",
        "i-SLOAN": "Sloan_i",
        "z-SLOAN": "Sloan_z",
        "g": "Sloan_g",
        "r": "Sloan_r",
        "i": "Sloan_i",
        "z": "Sloan_z",
        "B": "Johnson_B",
        "V": "Johnson_V",
        "R": "Johnson_R",
        "I": "Johnson_I",
        # Add Gaia mappings
        "G": "G",
        "BP": "BP",
        "RP": "RP",
        # Add PanSTARRS mappings
        "gMeanPSFMag": "Sloan_g",
        "rMeanPSFMag": "Sloan_r",
        "iMeanPSFMag": "Sloan_i",
        "zMeanPSFMag": "Sloan_z",
        "yMeanPSFMag": "y",
        # what's in USNO-B
        "R1": "Johnson_R",
        "R2": "Johnson_R",
        "B1": "Johnson_B",
        "B2": "Johnson_B",
    }

    # Define filter systems
    JOHNSON_FILTERS = {
        "Johnson_B",
        "Johnson_V",
        "Johnson_R",
        "Johnson_I",
        "B",
        "V",
        "R",
        "I",
    }

    def get_available_filters(catalog_info):
        """Get available filters from catalog info"""
        if catalog_info is None:
            # Default to ATLAS filters if no catalog specified
            return set(FILTER_MAPPINGS.values())
        return set(catalog_info["filters"].keys())

    def map_filter_name(filter_name, available_filters):
        """Map filter name to standardized name if available"""
        mapped_name = FILTER_MAPPINGS.get(filter_name, filter_name)
        if mapped_name in available_filters:
            return mapped_name
        return None

    # Get available filters
    available_filters = get_available_filters(catalog_info)

    # Start with defaults
    fit_in_johnson = "AB"
    basemag = "Sloan_r"

    # Handle Johnson preference
    if options.johnson and "Johnson_V" in available_filters:
        basemag = "Johnson_V"
        fit_in_johnson = "Johnson"

    # Handle explicit base magnitude
    if options.basemag is not None:
        mapped_filter = map_filter_name(options.basemag, available_filters)
        if mapped_filter is not None:
            basemag = mapped_filter
            if mapped_filter in JOHNSON_FILTERS:
                fit_in_johnson = "Johnson"
        else:
            logging.warning(
                f"Requested filter {options.basemag} not available in catalog. Using {basemag}"
            )

    # Handle automatic filter detection
    elif options.guessbase:
        filter_name = det.meta.get("FILTER")
        if filter_name:
            mapped_filter = map_filter_name(filter_name, available_filters)
            if mapped_filter is not None:
                basemag = mapped_filter
                if mapped_filter in JOHNSON_FILTERS:
                    fit_in_johnson = "Johnson"

    # Handle case where selected filter is not available
    if basemag not in available_filters:
        # Try to find best alternative
        if fit_in_johnson == "Johnson":
            alternatives = JOHNSON_FILTERS & available_filters
            if alternatives:
                basemag = next(iter(alternatives))
            else:
                # Fall back to Sloan if no Johnson filters available
                basemag = (
                    "Sloan_r"
                    if "Sloan_r" in available_filters
                    else next(iter(available_filters))
                )
                fit_in_johnson = "AB"
        else:
            # Try to find closest Sloan filter
            sloan_order = ["Sloan_r", "Sloan_g", "Sloan_i", "Sloan_z"]
            for alt in sloan_order:
                if alt in available_filters:
                    basemag = alt
                    break
            else:
                # Last resort: use first available filter
                basemag = next(iter(available_filters))

    logging.info(
        f"basemag={basemag} fit_in_johnson={fit_in_johnson} (available filters: {sorted(available_filters)})"
    )
    return basemag, fit_in_johnson


def x_get_base_filter(det, options):
    """set up or guess the base filter to use for fitting"""

    johnson_filters = [
        "Johnson_B",
        "Johnson_V",
        "Johnson_R",
        "Johnson_I",
        "B",
        "V",
        "R",
        "I",
    ]

    fit_in_johnson = "AB"
    basemag = "Sloan_r"

    if options.johnson:
        basemag = "Johnson_V"
        fit_in_johnson = "Johnson"

    if options.guessbase:
        if det.meta["FILTER"] == "Sloan_g":
            basemag = "Sloan_g"
        if det.meta["FILTER"] == "Sloan_r":
            basemag = "Sloan_r"
        if det.meta["FILTER"] == "Sloan_i":
            basemag = "Sloan_i"
        if det.meta["FILTER"] == "Sloan_z":
            basemag = "Sloan_z"
        if det.meta["FILTER"] == "g-SLOAN":
            basemag = "Sloan_g"
        if det.meta["FILTER"] == "r-SLOAN":
            basemag = "Sloan_r"
        if det.meta["FILTER"] == "i-SLOAN":
            basemag = "Sloan_i"
        if det.meta["FILTER"] == "z-SLOAN":
            basemag = "Sloan_z"
        if det.meta["FILTER"] == "Johnson_B":
            basemag = "Johnson_B"
        if det.meta["FILTER"] == "Johnson_V":
            basemag = "Johnson_V"
        if det.meta["FILTER"] == "Johnson_R":
            basemag = "Johnson_R"
        if det.meta["FILTER"] == "Johnson_I":
            basemag = "Johnson_I"
        if det.meta["FILTER"] == "g":
            basemag = "Sloan_g"
        if det.meta["FILTER"] == "r":
            basemag = "Sloan_r"
        if det.meta["FILTER"] == "i":
            basemag = "Sloan_i"
        if det.meta["FILTER"] == "z":
            basemag = "Sloan_z"
        if det.meta["FILTER"] == "B":
            basemag = "Johnson_B"
        if det.meta["FILTER"] == "V":
            basemag = "Johnson_V"
        if det.meta["FILTER"] == "R":
            basemag = "Johnson_R"
        if det.meta["FILTER"] == "I":
            basemag = "Johnson_I"

        if basemag in johnson_filters:
            fit_in_johnson = "Johnson"

    if options.basemag is not None:
        if options.basemag in johnson_filters:
            fit_in_johnson = "Johnson"
        basemag = options.basemag

    logging.info(f"basemag={basemag} fit_in_johnson={fit_in_johnson}")
    return basemag, fit_in_johnson

>>>>>>> cf38f76 (Add the daemon to transient analyser)

def write_stars_file(data, ffit, imgwcs, filename="stars"):
    """
    Write star data to a file for visualization purposes.

    Parameters:
    data : PhotometryData
        The PhotometryData object containing all star data
    ffit : object
        The fitting object containing the model and fit results
    filename : str, optional
        The name of the output file (default is "stars")
    """
    # Ensure we're using the most recent mask (likely the combined photometry and astrometry mask)
    # current_mask = data.get_current_mask()
    current_mask = data.get_current_mask()
    data.use_mask("default")

    # Get all required arrays
<<<<<<< HEAD
    fd_plot = data.get_fitdata('x', 'y', 'adif', 'coord_x', 'coord_y', 'color1', 'color2', 'color3', 'color4', 'img', 'dy', 'ra', 'dec', 'image_x', 'image_y', 'cat_x', 'cat_y', 'airmass')

    # Calculate model magnitudes using fotparams
    model_mags = ffit.model(np.array(ffit.fitvalues), fd_plot.fotparams)

    # Calculate astrometric residuals (if available)
    try:
        astx, asty = imgwcs.all_world2pix(fd_plot.ra, fd_plot.dec, 1)
        ast_residuals = np.sqrt((astx - fd_plot.coord_x)**2 + (asty - fd_plot.coord_y)**2)
=======
    (
        x,
        y,
        adif,
        coord_x,
        coord_y,
        color1,
        color2,
        color3,
        color4,
        img,
        dy,
        ra,
        dec,
        image_x,
        image_y,
    ) = data.get_arrays(
        "x",
        "y",
        "adif",
        "coord_x",
        "coord_y",
        "color1",
        "color2",
        "color3",
        "color4",
        "img",
        "dy",
        "ra",
        "dec",
        "image_x",
        "image_y",
    )

    # Calculate model magnitudes
    model_input = (
        y,
        adif,
        coord_x,
        coord_y,
        color1,
        color2,
        color3,
        color4,
        img,
        x,
        dy,
    )
    model_mags = ffit.model(np.array(ffit.fitvalues), model_input)

    # Calculate astrometric residuals (if available)
    try:
        astx, asty = imgwcs.all_world2pix(ra, dec, 1)
        ast_residuals = np.sqrt((astx - coord_x) ** 2 + (asty - coord_y) ** 2)
>>>>>>> cf38f76 (Add the daemon to transient analyser)
    except KeyError:
        ast_residuals = np.zeros_like(fd_plot.x)  # If astrometric data is not available

    # Create a table with all the data
<<<<<<< HEAD
    stars_table = Table([
        fd_plot.x, fd_plot.adif, fd_plot.image_x, fd_plot.image_y, fd_plot.color1, fd_plot.color2, fd_plot.color3, fd_plot.color4,
        model_mags, fd_plot.dy, fd_plot.ra, fd_plot.dec, astx, asty, ast_residuals, current_mask, current_mask, current_mask, fd_plot.cat_x, fd_plot.cat_y
    ], names=[
        'cat_mags', 'airmass', 'image_x', 'image_y', 'color1', 'color2', 'color3', 'color4',
        'model_mags', 'mag_err', 'ra', 'dec', 'ast_x', 'ast_y', 'ast_residual', 'mask', 'mask2', 'mask3', 'cat_x', 'cat_y'
    ])

    # Add column descriptions
    stars_table['cat_mags'].description = 'Catalog magnitude'
    stars_table['airmass'].description = 'Airmass difference from mean'
    stars_table['image_x'].description = 'X coordinate in image'
    stars_table['image_y'].description = 'Y coordinate in image'
    stars_table['color1'].description = 'Color index 1'
    stars_table['color2'].description = 'Color index 2'
    stars_table['color3'].description = 'Color index 3'
    stars_table['color4'].description = 'Color index 4'
    stars_table['model_mags'].description = 'Observed magnitude'
    stars_table['mag_err'].description = 'Magnitude error'
    stars_table['ra'].description = 'Right Ascension'
    stars_table['dec'].description = 'Declination'
    stars_table['ast_x'].description = 'Model position X'
    stars_table['ast_y'].description = 'Model position Y'
    stars_table['ast_residual'].description = 'Astrometric residual'
    stars_table['mask'].description = 'Boolean mask (True for included points)'
    stars_table['mask2'].description = 'Boolean mask (True for included points)'
    stars_table['mask3'].description = 'Boolean mask (True for included points)'
    stars_table['cat_x'].description = 'X coordinate in catalog'
    stars_table['cat_y'].description = 'Y coordinate in catalog'
=======
    stars_table = astropy.table.Table(
        [
            x,
            adif,
            image_x,
            image_y,
            color1,
            color2,
            color3,
            color4,
            model_mags,
            dy,
            ra,
            dec,
            astx,
            asty,
            ast_residuals,
            current_mask,
            current_mask,
            current_mask,
        ],
        names=[
            "cat_mags",
            "airmass",
            "image_x",
            "image_y",
            "color1",
            "color2",
            "color3",
            "color4",
            "model_mags",
            "mag_err",
            "ra",
            "dec",
            "ast_x",
            "ast_y",
            "ast_residual",
            "mask",
            "mask2",
            "mask3",
        ],
    )

    # Add column descriptions
    stars_table["cat_mags"].description = "Catalog magnitude"
    stars_table["airmass"].description = "Airmass difference from mean"
    stars_table["image_x"].description = "X coordinate in image"
    stars_table["image_y"].description = "Y coordinate in image"
    stars_table["color1"].description = "Color index 1"
    stars_table["color2"].description = "Color index 2"
    stars_table["color3"].description = "Color index 3"
    stars_table["color4"].description = "Color index 4"
    stars_table["model_mags"].description = "Observed magnitude"
    stars_table["mag_err"].description = "Magnitude error"
    stars_table["ra"].description = "Right Ascension"
    stars_table["dec"].description = "Declination"
    stars_table["ast_x"].description = "Model position X"
    stars_table["ast_y"].description = "Model position Y"
    stars_table["ast_residual"].description = "Astrometric residual"
    stars_table["mask"].description = "Boolean mask (True for included points)"
    stars_table["mask2"].description = "Boolean mask (True for included points)"
    stars_table["mask3"].description = "Boolean mask (True for included points)"
>>>>>>> cf38f76 (Add the daemon to transient analyser)

    # Write the table to a file
    stars_table.write(filename, format="ascii.ecsv", overwrite=True)


def perform_photometric_fitting(data, options, metadata):
    """
    Perform photometric fitting on the data using forward stepwise regression,
    with proper handling of initial values from loaded models.

    Args:
    data (PhotometryData): Object containing all photometry data.
    options (argparse.Namespace): Command line options.
    metadata (list): Metadata for the images being processed.

    Returns:
    fotfit.fotfit: The fitted photometry model.
    """
    # Set initial filter from already-determined metadata (from main loop)
    data.set_current_filter(metadata[0]['PHFILTER'])
    photometric_system = metadata[0]['PHSYSTEM']

<<<<<<< HEAD

    # Unified zeropoint computation and filter validation/discovery
    zeropoints, final_filter, filter_results = compute_zeropoints_all_filters(data, metadata, options)
=======
    #    fits_filter, photometric_system = get_base_filter(det, options)
    data.set_current_filter(metadata[0]["REFILTER"])
    photometric_system = metadata[0]["REJC"]

    if options.select_best:
        # First, try fitting zeropoints to each available catalog filter
        best_filter = select_best_filter(data, metadata)
        print(f"Selected best filter for initial calibration: {best_filter}")
        # Set the current filter in the PhotometryData object
        data.set_current_filter(best_filter)
        # Determine the photometric system based on the best filter
        photometric_system = "Johnson" if best_filter.startswith("Johnson") else "AB"
>>>>>>> cf38f76 (Add the daemon to transient analyser)

    # Update photometric system and schema if filter changed
    if final_filter != metadata[0]['PHFILTER']:
        try:
            from filter_matching import get_catalog_filters, find_compatible_schema
            catalog_name = 'makak' if options.makak else (options.catalog or 'atlas@localhost')
            available_filters = get_catalog_filters(catalog_name)

            if final_filter in available_filters:
                photometric_system = available_filters[final_filter].system
                # Find new compatible schema
                new_schema = find_compatible_schema(available_filters, options.filter_schemas, final_filter)
                if new_schema:
                    metadata[0]['PHSCHEMA'] = new_schema
            else:
                photometric_system = 'AB'  # Default fallback
        except:
            photometric_system = 'AB'  # Safe fallback

        # Update metadata for all images (discover mode finds one filter for entire dataset)
        for meta in metadata:
            meta['PHFILTER'] = final_filter
            meta['PHSYSTEM'] = photometric_system
            if 'new_schema' in locals() and new_schema:
                meta['PHSCHEMA'] = new_schema

    # Check for filter consistency across all images (after any corrections)
    if len(metadata) > 1:
        filter_list = [meta['PHFILTER'] for meta in metadata]
        unique_filters = set(filter_list)
        if len(unique_filters) > 1:
            print(f"ERROR: Mixed filters detected after validation: {sorted(unique_filters)}")
            filter_counts = {}
            for f in filter_list:
                filter_counts[f] = filter_counts.get(f, 0) + 1
            print(f"Filter distribution: {filter_counts}")
            print("Cannot fit mixed filters in single run. Process images separately.")
            sys.exit(1)

    # Compute colors and apply color limits
    data.compute_colors_and_apply_limits(metadata[0]['PHSCHEMA'], options)

    # Initialize fitting object
    ffit = fotfit.fotfit(fit_xy=options.fit_xy)

    # Note: ffit.zero is obsolete - zeropoints are now handled as regular terms (Z or Z:1, Z:2, etc.)

    # Create a dictionary to store initial values from loaded model
    initial_values = {}
    if options.model:
        ffit.readmodel(options.model)
        # Store initial values from the loaded model
        for term, value in zip(ffit.fixterms + ffit.fitterms,
                             ffit.fixvalues + ffit.fitvalues):
            initial_values[term] = value
        print(f"Model imported from {options.model}")
        print(f"Loaded initial values from model for terms: {list(initial_values.keys())}")

    # Load from RESPONSE header if no model file and single image
    elif len(metadata) == 1 and 'RESPONSE' in metadata[0]:
        response_string = metadata[0]['RESPONSE']
        # Parse RESPONSE manually to get initial values and put them in ffit for stepwise to see
        for chunk in response_string.split(","):
            if '=' not in chunk:
                continue
            term, strvalue = chunk.split("=")
            if term in ['FILTER', 'SCHEMA']:
                continue
            try:
                value = float(strvalue)
                initial_values[term] = value
                # Don't add to ffit - let the unified system handle all terms identically
            except ValueError:
                continue
        print(f"Loaded initial values from RESPONSE header for terms: {list(initial_values.keys())}")
        print(f"RESPONSE: {response_string}")

    # Perform initial fit with just the zeropoints
<<<<<<< HEAD
    fd = data.get_fitdata('y', 'adif', 'coord_x', 'coord_y', 'color1', 'color2', 'color3', 'color4', 'img', 'x', 'dy', 'image_x', 'image_y', 'airmass')

    # Add zeropoint terms to initial values only if not already present
    if options.single_zeropoint:
        if 'Z' not in initial_values:
            avg_zp = np.mean(zeropoints)
            initial_values['Z'] = avg_zp
            print(f"Single zeropoint mode: Using computed Z={avg_zp:.6f} as initial value")
=======
    fdata = data.get_arrays(
        "y",
        "adif",
        "coord_x",
        "coord_y",
        "color1",
        "color2",
        "color3",
        "color4",
        "img",
        "x",
        "dy",
    )
    ffit.fit(fdata)
    best_wssrndf = ffit.wssrndf

    # Parse the terms from options.terms
    all_terms = parse_terms(options.terms) if options.terms else []

    selected_terms = []
    remaining_terms = all_terms.copy()

    while remaining_terms:
        best_term = None
        best_improvement = 0

        # Try each remaining term in parallel
        with concurrent.futures.ProcessPoolExecutor() as executor:
            future_to_term = {
                executor.submit(
                    try_term, ffit, term, selected_terms, fdata, options
                ): term
                for term in remaining_terms
            }
            for future in concurrent.futures.as_completed(future_to_term):
                term = future_to_term[future]
                try:
                    new_wssrndf = future.result()
                    improvement = best_wssrndf - new_wssrndf
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_term = term
                except Exception as exc:
                    print(f"{term} generated an exception: {exc}")

        if best_improvement > 0:
            selected_terms.append(best_term)
            remaining_terms.remove(best_term)
            best_wssrndf -= best_improvement
            print(f"Added term {best_term}. New wssrndf: {best_wssrndf}")
>>>>>>> cf38f76 (Add the daemon to transient analyser)
        else:
            print(f"Single zeropoint mode: Using loaded Z={initial_values['Z']:.6f}")
    else:
        # Use per-image zeropoints
        for i, zp in enumerate(zeropoints, 1):
            zp_term = f"Z:{i}"
            if zp_term not in initial_values:
                initial_values[zp_term] = zp
        loaded_zp_terms = [k for k in initial_values.keys() if k.startswith('Z:')]
        computed_zp_terms = [f"Z:{i}" for i in range(1, len(zeropoints) + 1)]
        print(f"Multi-image zeropoints: computed {computed_zp_terms}, loaded {loaded_zp_terms}")

<<<<<<< HEAD
    # Build terms string starting with zeropoint terms
    terms_parts = []
    if options.single_zeropoint:
        terms_parts.append("&Z")  # Always direct (forced) term
    else:
        # Add per-image zeropoint terms
        zp_terms = [f"&Z:{i}" for i in range(1, len(zeropoints) + 1)]
        terms_parts.extend(zp_terms)
=======
    # Final fit with all selected terms
    setup_photometric_model(ffit, options, trying=",".join(selected_terms))
    ffit.fit(fdata)
    # Apply photometric mask and refit
    photo_mask = ffit.residuals(ffit.fitvalues, fdata) < 5 * ffit.wssrndf
    data.apply_mask(photo_mask, "photometry")
    data.use_mask("photometry")
    # 1.
    fdata = data.get_arrays(
        "y",
        "adif",
        "coord_x",
        "coord_y",
        "color1",
        "color2",
        "color3",
        "color4",
        "img",
        "x",
        "dy",
    )
>>>>>>> cf38f76 (Add the daemon to transient analyser)

    # Add per-image PX/PY terms if fit_xy is enabled (compact form)
    if options.fit_xy:
        terms_parts.append("&*.p")  # Expands to &*PX,&*PY automatically

<<<<<<< HEAD
    # Add user terms if specified
    if options.terms:
        terms_parts.append(options.terms)

    extended_terms = ",".join(terms_parts)

    # Parse extended term syntax with auto-added zeropoints
    parsed_terms = parse_terms(extended_terms, len(metadata))
    print(f"Extended terms string: {extended_terms}")
    print(f"Parsed terms: stepwise={parsed_terms['stepwise']}, direct={parsed_terms['direct']}, "
          f"fixed={list(parsed_terms['fixed'].keys())}, default={parsed_terms['default']}")

    # Merge initial values from model file and command line
    combined_initial_values = {**initial_values, **parsed_terms['initial_values']}

    # Determine term assignment based on --use-stepwise default and per-term modifiers
    if options.use_stepwise:
        # Default behavior is stepwise for unmarked terms
        stepwise_terms = parsed_terms['stepwise'] + parsed_terms['default']
        direct_terms = parsed_terms['direct']
        print(f"Default: stepwise. Stepwise terms: {stepwise_terms}, Direct terms: {direct_terms}")
    else:
        # Default behavior is direct for unmarked terms
        stepwise_terms = parsed_terms['stepwise']
        direct_terms = parsed_terms['direct'] + parsed_terms['default']
        print(f"Default: direct. Stepwise terms: {stepwise_terms}, Direct terms: {direct_terms}")

    # Set up fixed terms first
    ffit.fixall()  # Reset all terms
    if parsed_terms['fixed']:
        fixed_terms = list(parsed_terms['fixed'].keys())
        fixed_values = list(parsed_terms['fixed'].values())
        print(f"Setting fixed terms: {dict(zip(fixed_terms, fixed_values))}")
        ffit.fixterm(fixed_terms, values=fixed_values)

    # Handle the unified fitting approach
    print(f"Using unified stepwise approach:")
    print(f"  - Always selected (direct): {direct_terms}")
    print(f"  - Stepwise candidates: {stepwise_terms}")

    # Pass combined initial values to stepwise regression
    options._combined_initial_values = combined_initial_values

    # Perform stepwise regression with always-selected direct terms
    # This handles all cases: with stepwise terms, without stepwise terms, or only direct terms
    selected_stepwise_terms, final_wssrndf = perform_stepwise_regression(
        data, ffit, stepwise_terms, options, metadata, always_selected=direct_terms
    )

    # The result already includes both direct terms (always selected) and stepwise terms
    selected_terms = selected_stepwise_terms

    print(f"Final fit variance: {ffit.wssrndf}")
    if selected_terms:
        # Filter out Z:n terms from grouped display (they're shown in 2D table)
        non_z_terms = [t for t in selected_terms if not (t.startswith('Z:') or t == 'Z')]
    else:
        print("No terms selected")

    return ffit

=======
    # Final fit with refined mask
    data.use_mask("default")
    fdata = data.get_arrays(
        "y",
        "adif",
        "coord_x",
        "coord_y",
        "color1",
        "color2",
        "color3",
        "color4",
        "img",
        "x",
        "dy",
    )
    photo_mask = ffit.residuals(ffit.fitvalues, fdata) < 5 * ffit.wssrndf
    data.apply_mask(photo_mask, "photometry")
    data.use_mask("photometry")
    # 1.
    fdata = data.get_arrays(
        "y",
        "adif",
        "coord_x",
        "coord_y",
        "color1",
        "color2",
        "color3",
        "color4",
        "img",
        "x",
        "dy",
    )
    ffit.fit(fdata)

    print(f"Final fit variance: {ffit.wssrndf}")

    print(f"Final fit variance: {ffit.wssrndf}")
    print(f"Selected terms: {selected_terms}")

    return ffit


def try_term(ffit, term, selected_terms, fdata, options):
    """
    Try fitting with a new term added to the currently selected terms.

    Args:
    ffit (fotfit.fotfit): The current fitting object.
    term (str): The new term to try.
    selected_terms (list): List of currently selected terms.
    fdata (tuple): The fitting data.

    Returns:
    float: The new wssrndf after fitting with the added term.
    """
    new_ffit = deepcopy(ffit)
    setup_photometric_model(new_ffit, options, trying=",".join(selected_terms + [term]))
    new_ffit.fit(fdata)
    return new_ffit.wssrndf


def parse_terms(terms_string):
    """
    Parse the terms string into a list of individual terms.

    Args:
    terms_string (str): Comma-separated string of terms.

    Returns:
    list: List of individual terms.
    """
    return [term.strip() for term in terms_string.split(",") if term.strip()]


def setup_photometric_model(ffit, options, trying=None):
    """
    Set up the photometric model based on command line options.

    Args:
    ffit (fotfit.fotfit): The photometric fitting object.
    options (argparse.Namespace): Command line options.
    """

    # Load model from file if specified
    if options.model:
        load_model_from_file(ffit, options.model)

    ffit.fixall()  # Fix all terms initially
    ffit.fixterm(["N1"], values=[0])

    # Set up fitting terms
    if trying is None:
        if options.terms:
            setup_fitting_terms(ffit, options.terms, options.verbose, options.fit_xy)
    else:
        setup_fitting_terms(ffit, trying, options.verbose, options.fit_xy)


>>>>>>> cf38f76 (Add the daemon to transient analyser)
def load_model_from_file(ffit, model_file):
    """
    Load a photometric model from a file.

    Args:
    ffit (fotfit.fotfit): The photometric fitting object.
    model_file (str): Path to the model file.
    """
    model_paths = [
        model_file,
<<<<<<< HEAD
        f"/home/mates/pyrt/model/{model_file}.mod"
        #f"/home/mates/pyrt/model/{model_file}-{ffit.det.meta['FILTER']}.mod"
=======
        f"/home/mates/pyrt/model/{model_file}.mod",
        f"/home/mates/pyrt/model/{model_file}-{ffit.det.meta['FILTER']}.mod",
>>>>>>> cf38f76 (Add the daemon to transient analyser)
    ]
    for path in model_paths:
#        try:
            ffit.readmodel(path)
            print(f"Model imported from {path}")
            return
#        except:
#            pass
    print(f"Cannot open model {model_file}")

<<<<<<< HEAD
def update_det_file(fitsimgf: str, options): #  -> Tuple[Optional[Table], str]:
=======

def setup_fitting_terms(ffit, terms, verbose, fit_xy):
    """
    Set up fitting terms based on the provided options.

    Args:
    ffit (fotfit.fotfit): The photometric fitting object.
    terms (str): Comma-separated list of terms to fit.
    verbose (bool): Whether to print verbose output.
    fit_xy (bool): Whether to fit xy tilt for each image separately.
    """
    for term in terms.split(","):
        if term == "":
            continue  # Skip empty terms
        if term[0] == ".":
            setup_polynomial_terms(ffit, term, verbose, fit_xy)
        else:
            ffit.fitterm([term], values=[1e-6])


def setup_polynomial_terms(ffit, term, verbose, fit_xy):
    """
    Set up polynomial terms for fitting.

    Args:
    ffit (fotfit.fotfit): The photometric fitting object.
    term (str): The polynomial term descriptor.
    verbose (bool): Whether to print verbose output.
    fit_xy (bool): Whether to fit xy tilt for each image separately.
    """
    if term[1] == "p":
        setup_surface_polynomial(ffit, term, verbose, fit_xy)
    elif term[1] == "r":
        setup_radial_polynomial(ffit, term, verbose)


def setup_surface_polynomial(ffit, term, verbose, fit_xy):
    """Set up surface polynomial terms."""
    pol_order = int(term[2:])
    if verbose:
        print(f"Setting up a surface polynomial of order {pol_order}:")
    polytxt = "P(x,y)="
    for pp in range(1, pol_order + 1):
        if fit_xy and pp == 1:
            continue
        for rr in range(0, pp + 1):
            polytxt += f"+P{rr}X{pp-rr}Y*x**{rr}*y**{pp-rr}"
            ffit.fitterm([f"P{rr}X{pp-rr}Y"], values=[1e-6])
    if verbose:
        print(polytxt)


def setup_radial_polynomial(ffit, term, verbose):
    """Set up radial polynomial terms."""
    pol_order = int(term[2:])
    if verbose:
        print(f"Setting up a polynomial of order {pol_order} in radius:")
    polytxt = "P(r)="
    for pp in range(1, pol_order + 1):
        polytxt += f"+P{pp}R*(x**2+y**2)**{pp}"
        ffit.fitterm([f"P{pp}R"], values=[1e-6])
    if verbose:
        print(polytxt)


def update_det_file(
    fitsimgf: str, options
):  #  -> Tuple[Optional[astropy.table.Table], str]:
>>>>>>> cf38f76 (Add the daemon to transient analyser)
    """Update the .det file using cat2det.py."""
    cmd = ["cat2det.py"]
    if options.verbose:
        cmd.append("-v")
    if options.filter:
        cmd.extend(["-f", options.filter])
    cmd.append(fitsimgf)

    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running cat2det.py: {e}")
        return None, ""

    return try_det(os.path.splitext(fitsimgf)[0] + ".det", options.verbose)


def should_update_det_file(det, fitsimg, detf, fitsimgf, options):
    """Determine if the .det file should be updated."""
    if det is None and fitsimg is not None:
        return True
    if det is not None and fitsimg is not None and options.autoupdate:
        return os.path.getmtime(fitsimgf) > os.path.getmtime(detf)
    return False

<<<<<<< HEAD
def apply_spatial_correction(det, model_string):
    """
    Apply model correction using the same approach as mesh_flat
    """
    ffit = fotfit.fotfit(fit_xy=False)
    ffit.from_oneline(model_string)
=======

def process_input_file(arg, options):
>>>>>>> cf38f76 (Add the daemon to transient analyser)

    # Add original zeropoint to fitvalues
    #ffit.fitvalues = np.concatenate([ffit.fitvalues, [det.meta['MAGZERO']]])
    ffit.fitvalues = np.concatenate([ffit.fitvalues, [0.0]])

    data = (
        np.zeros_like(det['MAG_AUTO']),  # mc
        det.meta.get('AIRMASS', 1.0),    # airmass
        (det['X_IMAGE'] - det.meta['CTRX'])/1024,  # normalized X
        (det['Y_IMAGE'] - det.meta['CTRY'])/1024,  # normalized Y
        np.zeros_like(det['MAG_AUTO']),  # color1
        np.zeros_like(det['MAG_AUTO']),  # color2
        np.zeros_like(det['MAG_AUTO']),  # color3
        np.zeros_like(det['MAG_AUTO']),  # color4
        det.meta.get('IMGNO', 0),
        np.zeros_like(det['MAG_AUTO']),  # y
        np.ones_like(det['MAG_AUTO']),   # err
        det['X_IMAGE'],                  # raw X
        det['Y_IMAGE']                   # raw Y
    )

    correction = ffit.model(ffit.fitvalues, data)
    logging.info(f"Correction stats - min: {np.min(correction):.3f}, max: {np.max(correction):.3f}, mean: {np.mean(correction):.3f}")
    det['MAG_AUTO'] += correction

    return det

def process_input_file(arg, options):
    """Modified to handle spatial detrending"""
    detf, det = try_det(arg, options.verbose)
    if det is None:
        detf, det = try_det(os.path.splitext(arg)[0] + ".det", options.verbose)
    if det is None:
        detf, det = try_det(arg + ".det", options.verbose)

    fitsimgf, fitsimg = try_img(arg + ".fits", options.verbose)
    if fitsimg is None:
        fitsimgf, fitsimg = try_img(os.path.splitext(arg)[0] + ".fits", options.verbose)
    if fitsimg is None:
        fitsimgf, fitsimg = try_img(os.path.splitext(arg)[0], options.verbose)

    # with these, we should have the possible filenames covered, now lets see what we got
    # 1. have fitsimg, no .det: call cat2det, goto 3
    # 2. have fitsimg, have det, det is older than fitsimg: same as 1
    if should_update_det_file(det, fitsimg, detf, fitsimgf, options):
        detf, det = update_det_file(fitsimgf, options)
        logging.info("Back in dophot")

    if det is None:
        return None
    # print(type(det.meta)) # OrderedDict, so we can:
<<<<<<< HEAD
    with suppress(KeyError, AttributeError): det.meta.pop('comments')
    with suppress(KeyError, AttributeError): det.meta.pop('history')

    # 3. have fitsimg, have .det, note the fitsimg filename, close fitsimg and run
    # 4. have det, no fitsimg: just run, writing results into fits will be disabled
    if fitsimgf is not None: det.meta['filename'] = fitsimgf
    det.meta['detf'] = detf
    # Apply spatial correction if requested
    if options.remove_spatial and 'RESPONSE' in det.meta:
        det = apply_spatial_correction(det, det.meta['RESPONSE'])
        logging.info("Applied spatial correction from existing model")
=======
    with suppress(KeyError):
        det.meta.pop("comments")
    with suppress(KeyError):
        det.meta.pop("history")

    # 3. have fitsimg, have .det, note the fitsimg filename, close fitsimg and run
    # 4. have det, no fitsimg: just run, writing results into fits will be disabled
    if fitsimgf is not None:
        det.meta["filename"] = fitsimgf
    det.meta["detf"] = detf
    logging.info(f"DetF={detf}, ImgF={fitsimgf}")
>>>>>>> cf38f76 (Add the daemon to transient analyser)

    logging.info(f"DetF={detf}, ImgF={fitsimgf}")
    return det


def setup_logging(verbose):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


    # Silence matplotlib's debug spam
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

def write_results(data, ffit, options, alldet, target, zpntest):
    zero, zerr = ffit.zero_val()

    for img, det in enumerate(alldet):
        zero_idx = min(img, len(zero) - 1)
        zerr_idx = min(img, len(zerr) - 1)

        start = time.time()
        if options.astrometry:
            try:
<<<<<<< HEAD
                astropy.io.fits.setval(os.path.splitext(det.meta['FITSFILE'])[0]+"t.fits", "LIMMAG", 0, value=zero[zero_idx]+det.meta['LIMFLX3'])
                astropy.io.fits.setval(os.path.splitext(det.meta['FITSFILE'])[0]+"t.fits", "MAGZERO", 0, value=zero[zero_idx])
                astropy.io.fits.setval(os.path.splitext(det.meta['FITSFILE'])[0]+"t.fits", "RESPONSE", 0, value=ffit.oneline_for_image(img + 1))
=======
                astropy.io.fits.setval(
                    os.path.splitext(det.meta["FITSFILE"])[0] + "t.fits",
                    "LIMMAG",
                    0,
                    value=zero[img] + det.meta["LIMFLX3"],
                )
                astropy.io.fits.setval(
                    os.path.splitext(det.meta["FITSFILE"])[0] + "t.fits",
                    "MAGZERO",
                    0,
                    value=zero[img],
                )
                astropy.io.fits.setval(
                    os.path.splitext(det.meta["FITSFILE"])[0] + "t.fits",
                    "RESPONSE",
                    0,
                    value=ffit.oneline(),
                )
>>>>>>> cf38f76 (Add the daemon to transient analyser)
            except Exception as e:
                logging.warning(
                    f"Writing LIMMAG/MAGZERO/RESPONSE to an astrometrized image failed: {e}"
                )
        logging.info(f"Writing to astrometrized image took {time.time()-start:.3f}s")

        start = time.time()
<<<<<<< HEAD
        # Only update RESPONSE if we're not using spatial correction
        fn = os.path.splitext(det.meta['detf'])[0] + ".ecsv"
        det['MAG_CALIB'] = ffit.model(np.array(ffit.fitvalues),
            (   det['MAG_AUTO'],     # magnitude
                0,                   # airmass (relative to image center = 0)
                det['X_IMAGE']/1024 - det.meta['CTRX']/1024, # X-coord
                det['Y_IMAGE']/1024 - det.meta['CTRY']/1024, # Y-coord
                0, 0, 0, 0, # colors (zeroes, we output instrumental mag)
                img,  # image index
                0, 0, # y and dy (obviously not used in model)
                det['X_IMAGE'], # x for pixel structure
                det['Y_IMAGE'], # y for pixel structure
                det.meta['AIRMASS']) # airmass (absolute)
            )
        det['MAGERR_CALIB'] = np.sqrt(np.power(det['MAGERR_AUTO'],2)+np.power(zerr[zerr_idx],2))

        # our zeropoint is a magnitude that 10000 counts are, motivation:
        # solving nonlinearity introduced poor fix for this value during fit
        det.meta['MAGZERO'] = zero[zero_idx]
        det.meta['DMAGZERO'] = zerr[zerr_idx]
        det.meta['MAGLIMIT'] = det.meta['LIMFLX3']+zero[zero_idx]  # Now using astronomical zeropoint directly
        det.meta['WSSRNDF'] = ffit.wssrndf
=======
        fn = os.path.splitext(det.meta["detf"])[0] + ".ecsv"
        det["MAG_CALIB"] = ffit.model(
            np.array(ffit.fitvalues),
            (
                det["MAG_AUTO"],
                det.meta["AIRMASS"],
                det["X_IMAGE"] / 1024 - det.meta["CTRX"] / 1024,
                det["Y_IMAGE"] / 1024 - det.meta["CTRY"] / 1024,
                0,
                0,
                0,
                0,
                img,
                0,
                0,
            ),
        )
        det["MAGERR_CALIB"] = np.sqrt(
            np.power(det["MAGERR_AUTO"], 2) + np.power(zerr[img], 2)
        )

        det.meta["MAGZERO"] = zero[img]
        det.meta["DMAGZERO"] = zerr[img]
        det.meta["MAGLIMIT"] = det.meta["LIMFLX3"] + zero[img]
        det.meta["WSSRNDF"] = ffit.wssrndf
        det.meta["RESPONSE"] = ffit.oneline()
>>>>>>> cf38f76 (Add the daemon to transient analyser)

        det.meta['RESPONSE'] = ffit.oneline_for_image(img + 1)

        if (zerr[zerr_idx] < 0.2) or (options.reject is None):
            det.write(fn, format="ascii.ecsv", overwrite=True)
        else:
            if options.verbose:
                logging.warning("rejected (too large uncertainty in zeropoint)")
            with open("rejected", "a+") as out_file:
<<<<<<< HEAD
                out_file.write(f"{det.meta['FITSFILE']} {ffit.wssrndf:.6f} {zerr[zerr_idx]:.3f}\n")
=======
                out_file.write(
                    f"{det.meta['FITSFILE']} {ffit.wssrndf:.6f} {zerr[img]:.3f}\n"
                )
>>>>>>> cf38f76 (Add the daemon to transient analyser)
            sys.exit(0)
        logging.info(f"Saving ECSV output took {time.time()-start:.3f}s")

        if options.flat or options.weight:
            start = time.time()
            ffData = mesh_create_flat_field_data(det, ffit)
            logging.info(f"Generating the flat field took {time.time()-start:.3f}s")
<<<<<<< HEAD
        if options.weight: save_weight_image(det, ffData, options, zpntest)
        if options.flat: save_flat_image(det, ffData, zpntest)
=======
        if options.weight:
            save_weight_image(det, ffData, options)
        if options.flat:
            save_flat_image(det, ffData)
>>>>>>> cf38f76 (Add the daemon to transient analyser)

        target_idx = min(img, len(target) - 1) if hasattr(target, '__len__') else img
        write_output_line(det, options, zero[zero_idx], zerr[zerr_idx], ffit, target[target_idx])


def mesh_create_flat_field_data(det, ffit):
    """
    Create the flat field data array.

    :param det: Detection table containing metadata
    :param ffit: Fitting object containing the flat field model
    :return: numpy array of flat field data
    """
    logging.info("Generating the flat-field array")
<<<<<<< HEAD
    shape = (det.meta['IMGAXIS2'], det.meta['IMGAXIS1'])
    y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')

    return ffit.mesh_flat(x, y,
                     ctrx=det.meta['CTRX'],
                     ctry=det.meta['CTRY'],
                     img=det.meta['IMGNO'])

def save_weight_image(det, ffData, options, zpntest):
=======
    shape = (det.meta["IMGAXIS2"], det.meta["IMGAXIS1"])
    y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")

    return ffit.mesh_flat(
        x, y, ctrx=det.meta["CTRX"], ctry=det.meta["CTRY"], img=det.meta["IMGNO"]
    )


def old_create_flat_field_data(det, ffit):
    """
    Create the flat field data array.
    
    :param det: Detection table containing metadata
    :param ffit: Fitting object containing the flat field model
    :return: numpy array of flat field data
    """
    logging.info("Generating the flat-field array")
    return np.fromfunction(
        ffit.old_flat,
        [det.meta["IMGAXIS2"], det.meta["IMGAXIS1"]],
        ctrx=det.meta["CTRX"],
        ctry=det.meta["CTRY"],
        img=det.meta["IMGNO"],
    )


def save_weight_image(det, ffData, options):
>>>>>>> cf38f76 (Add the daemon to transient analyser)
    """
    Generate and save the weight image.

    :param det: Detection table containing metadata
    :param ffData: Flat field data array
    :param options: Command line options
    """
    start = time.time()
    gain = options.gain if options.gain is not None else 2.3
    wwData = np.power(10, -0.4 * (22 - ffData))
    wwData = np.power(wwData * gain, 2) / (
        wwData * gain
        + 3.1315 * np.power(det.meta["BGSIGMA"] * gain * det.meta["FWHM"] / 2, 2)
    )
    wwHDU = astropy.io.fits.PrimaryHDU(data=wwData)
    weightfile = os.path.splitext(det.meta["filename"])[0] + "w.fits"

    if os.path.isfile(weightfile):
        logging.warning(f"Operation will overwrite an existing image: {weightfile}")
        os.unlink(weightfile)

    with astropy.io.fits.open(weightfile, mode="append") as wwFile:
        wwFile.append(wwHDU)

<<<<<<< HEAD
    if options.astrometry:
        zpntest.write(weightfile)
        astropy.io.fits.setval(os.path.splitext(det.meta['FITSFILE'])[0]+"t.fits", "WGHTFILE", 0, value=weightfile)

    logging.info(f"Weight image saved to {weightfile}")
    logging.info(f"Writing the weight file took {time.time()-start:.3f}s")

def save_flat_image(det, ffData, zpntest):
=======
    logging.info(f"Weight image saved to {weightfile}")
    logging.info(f"Writing the weight file took {time.time()-start:.3f}s")


def save_flat_image(det, ffData):
>>>>>>> cf38f76 (Add the daemon to transient analyser)
    """
    Generate and save the flat image.

    :param det: Detection table containing metadata
    :param ffData: Flat field data array
    """
    start = time.time()
    flatData = np.power(10, 0.4 * (ffData - 8.9))
    ffHDU = astropy.io.fits.PrimaryHDU(data=flatData)
    flatfile = os.path.splitext(det.meta["filename"])[0] + "f.fits"

    if os.path.isfile(flatfile):
        logging.warning(f"Operation will overwrite an existing image: {flatfile}")
        os.unlink(flatfile)

    with astropy.io.fits.open(flatfile, mode="append") as ffFile:
        ffFile.append(ffHDU)

<<<<<<< HEAD
    zpntest.write(flatfile)

    logging.info(f"Flat image saved to {flatfile}")
    logging.info(f"Writing the flatfield file took {time.time()-start:.3f}s")

def write_output_line(det, options, zero, zerr, ffit, target):
    tarid = det.meta.get('TARGET', 0)
    obsid = det.meta.get('OBSID', 0)
    if options.makak:
        tarid = det.meta.get('BGSIGMA', 0)
        obsid = det.meta.get('CCD_TEMP', 0)

    chartime = det.meta['JD'] + det.meta['EXPTIME'] / 2.0 / 86400.0
    if options.date == 'char':
        chartime = det.meta.get('CHARTIME', chartime)
    elif options.date == 'bjd':
        chartime = det.meta.get('BJD', chartime)

    if target is not None:
        tarx = target['X_IMAGE']/1024 - det.meta['CTRX']/1024
        tary = target['Y_IMAGE']/1024 - det.meta['CTRY']/1024
        data_target = np.array([[target['MAG_AUTO']], [0], [tarx], [tary], [0], [0], [0], [0], [det.meta['IMGNO']], [0], [0], [0.5], [0.5], [det.meta['AIRMASS']]])
=======
    logging.info(f"Flat image saved to {flatfile}")
    logging.info(f"Writing the flatfield file took {time.time()-start:.3f}s")


def write_output_line(det, options, zero, zerr, ffit, target):
    tarid = det.meta.get("TARGET", 0)
    obsid = det.meta.get("OBSID", 0)

    chartime = det.meta["JD"] + det.meta["EXPTIME"] / 2
    if options.date == "char":
        chartime = det.meta.get("CHARTIME", chartime)
    elif options.date == "bjd":
        chartime = det.meta.get("BJD", chartime)

    if target is not None:
        tarx = target["X_IMAGE"] / 1024 - det.meta["CTRX"] / 1024
        tary = target["Y_IMAGE"] / 1024 - det.meta["CTRY"] / 1024
        data_target = np.array(
            [
                [target["MAG_AUTO"]],
                [det.meta["AIRMASS"]],
                [tarx],
                [tary],
                [0],
                [0],
                [0],
                [0],
                [det.meta["IMGNO"]],
                [0],
                [0],
            ]
        )
>>>>>>> cf38f76 (Add the daemon to transient analyser)
        mo = ffit.model(np.array(ffit.fitvalues), data_target)

        out_line = f"{det.meta['FITSFILE']} {det.meta['JD']:.6f} {chartime:.6f} {det.meta['FILTER']} {det.meta['EXPTIME']:3.0f} {det.meta['AIRMASS']:6.3f} {det.meta['IDNUM']:4d} {zero:7.3f} {zerr:6.3f} {det.meta['LIMFLX3']+zero:7.3f} {ffit.wssrndf:6.3f} {mo[0]:7.3f} {target['MAGERR_AUTO']:6.3f} {tarid} {obsid} ok"
    else:
        out_line = f"{det.meta['FITSFILE']} {det.meta['JD']:.6f} {chartime:.6f} {det.meta['FILTER']} {det.meta['EXPTIME']:3.0f} {det.meta['AIRMASS']:6.3f} {det.meta['IDNUM']:4d} {zero:7.3f} {zerr:6.3f} {det.meta['LIMFLX3']+zero:7.3f} {ffit.wssrndf:6.3f}  99.999  99.999 {tarid} {obsid} not_found"

    print(out_line)

    try:
        with open("dophot.dat", "a") as out_file:
            out_file.write(out_line + "\n")
    except Exception as e:
        print(f"Error writing to dophot.dat: {e}")

<<<<<<< HEAD
=======

def select_best_filter(data, metadata):
    """
    Select the best catalog filter for initial calibration.

    Args:
    data (PhotometryData): Object containing all photometry data.
    metadata (list): List of metadata for each image.

    Returns:
    str: The name of the best filter to use for initial calibration.
    """
    available_filters = data.get_filter_columns()
    best_filter = None
    best_wssrndf = float("inf")

    for filter_name in available_filters:
        ffit = fotfit.fotfit()
        # Initialize zeropoints with zeros (we'll estimate them later)
        ffit.zero = [0] * len(metadata)

        # Temporarily set the current filter
        data.set_current_filter(filter_name)

        # Get the data for fitting, including placeholder color columns
        try:
            fdata = data.get_arrays("y", "adif", "coord_x", "coord_y", "img", "x", "dy")

            # Add placeholder color columns
            placeholder_colors = np.zeros((4, len(fdata[0])))
            fdata = fdata[:5] + tuple(placeholder_colors) + fdata[5:]

            ffit.fit(fdata)
            if ffit.wssrndf < best_wssrndf:
                best_wssrndf = ffit.wssrndf
                best_filter = filter_name
            print(
                f"Successfully fitted filter {filter_name} with wssrndf: {ffit.wssrndf}"
            )
        except Exception as e:
            print(f"Error fitting filter {filter_name}: {str(e)}")
            print(f"fdata shape: {[len(arr) for arr in fdata]}")
            print(f"First few elements of fdata: {[arr[:5] for arr in fdata]}")

    if best_filter is None:
        raise ValueError(
            "Failed to fit any filter. Check the input data and fitting process."
        )

    print(f"Best filter: {best_filter} with wssrndf: {best_wssrndf}")
    return best_filter
>>>>>>> cf38f76 (Add the daemon to transient analyser)


# ******** main() **********


def main():
    """Take over the world."""
    options = parse_arguments()
    setup_logging(options.verbose)

    logging.info(
        f"{os.path.basename(sys.argv[0])} running in Python {sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}"
    )
    logging.info(f"Magnitude limit set to {options.maglim}")

    data = PhotometryData()

    target = []
    metadata = []
    alldet = []
    imgno = 0

    for arg in options.files:

        det = process_input_file(arg, options)
        if det is None:
            logging.warning(f"I do not know what to do with {arg}")
            continue

<<<<<<< HEAD
        catalog_name = 'makak' if options.makak else (options.catalog or 'atlas@localhost')
        determine_filter(det, options, catalog_name)
        logging.info(f'Reference filter is {det.meta["PHFILTER"]}, '
                f'Schema: {det.meta["PHSCHEMA"]}, '
                f'System: {det.meta["PHSYSTEM"]}')

        start = time.time()

        cat, matches, imgwcs, target_match = process_image_with_dynamic_limits(det, options)
        if cat is None:
            logging.warning(f"Failed to process {arg}, skipping")
            continue

        logging.info(f"Catalog processing took {time.time()-start:.3f}s")
=======
        #    some_file = open("det.reg", "w+")
        #    some_file.write("# Region file format: DS9 version 4.1\nglobal color=red dashlist=8 3 width=3"\
        #        " font=\"helvetica 10 normal roman\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n")
        #    for w_ra, w_dec in zip(det['X_IMAGE'],det['Y_IMAGE']):
        #        some_file.write(f"circle({w_ra:.7f},{w_dec:.7f},3\") # color=red width=3\n")
        #    some_file.close()
>>>>>>> cf38f76 (Add the daemon to transient analyser)


        det["ALPHA_J2000"], det["DELTA_J2000"] = imgwcs.all_pix2world(
            [det["X_IMAGE"]], [det["Y_IMAGE"]], 1
        )

<<<<<<< HEAD
        logging.info("Reference filter is %s, Photometric schema: %s, Photometric system: %s"\
            %(det.meta['PHFILTER'],det.meta['PHSCHEMA'],det.meta['PHSYSTEM']))

        # make pairs to be fitted
        det.meta['IMGNO'] = imgno
        n_matched_stars = make_pairs_to_fit(det, cat, matches, imgwcs, options, data, None, target_match)
        det.meta['IDNUM'] = n_matched_stars
=======
        # if options.usewcs is not None:
        #      fixme from "broken_usewcs.py"

        start = time.time()

        # if options.makak_path: CatalogManager.set_makak_path(options.makak_path)
        catalog = "makak" if options.makak else (options.catalog or "atlas@localhost")
        enlarge = options.enlarge if options.enlarge is not None else 1.0

        logging.info(
            f"Catalog: {options.catalog} from command line, {catalog}  to be used."
        )

        cat = Catalog(
            ra=det.meta["CTRRA"],
            dec=det.meta["CTRDEC"],
            width=enlarge * det.meta["FIELD"],
            height=enlarge * det.meta["FIELD"],
            mlim=options.maglim,
            catalog=catalog,
        )

        if cat is None:
            logging.error(f"Failed to get {catalog} catalog data")
            return None

        catalog_info = Catalog.KNOWN_CATALOGS.get(catalog)
        base_filter, photometric_system = get_base_filter(det, options, catalog_info)

        # Epoch for PM correction (Gaia DR2@2015.5)
        # epoch = ( det.meta['JD'] - 2457204.5 ) / 365.2425 # Epoch for PM correction (Gaia DR2@2015.5)
        epoch = (
            (det.meta["JD"] - 2440587.5) - (cat.meta["astepoch"] - 1970.0) * 365.2425
        ) / 365.2425
        logging.info(f"EPOCH_DIFF={epoch}")
        cat["radeg"] += epoch * cat["pmra"]
        cat["decdeg"] += epoch * cat["pmdec"]

        logging.info(f"Catalog search took {time.time()-start:.3f}s")
        logging.info(f"Catalog contains {len(cat)} entries")

        # write_region_file(cat, "cat.reg")

        if options.idlimit:
            idlimit = options.idlimit
        else:
            try:
                idlimit = det.meta["FWHM"]
                logging.info(
                    f"idlimit set to fits header FWHM value of {idlimit} pixels."
                )
            except KeyError:
                idlimit = 2.0 / 3600
                logging.info(
                    f"idlimit set to a hard-coded default of {idlimit} pixels."
                )

        # ===  identification with KDTree  ===
        Y = np.array([det["X_IMAGE"], det["Y_IMAGE"]]).transpose()

        start = time.time()
        try:
            Xt = np.array(imgwcs.all_world2pix(cat["radeg"], cat["decdeg"], 1))
        except (ValueError, KeyError):
            logging.warning(f"Astrometry of {arg} sucks! Skipping.")
            continue

        # clean up the array from off-image objects
        cat = cat[
            np.logical_not(
                np.any(
                    [
                        np.isnan(Xt[0]),
                        np.isnan(Xt[1]),
                        Xt[0] < 0,
                        Xt[1] < 0,
                        Xt[0] > det.meta["IMGAXIS1"],
                        Xt[1] > det.meta["IMGAXIS2"],
                    ],
                    axis=0,
                )
            )
        ]

        Xt = np.array(imgwcs.all_world2pix(cat["radeg"], cat["decdeg"], 1))
        X = Xt.transpose()

        tree = KDTree(X)
        nearest_ind, nearest_dist = tree.query_radius(
            Y, r=idlimit, return_distance=True, count_only=False
        )
        logging.info(f"Object cross-id took {time.time()-start:.3f}s")

        # identify the target object
        target.append(None)
        if det.meta["OBJRA"] < -99 or det.meta["OBJDEC"] < -99:
            logging.info("Target was not defined")
        else:
            logging.info(
                f"Target coordinates: {det.meta['OBJRA']:.6f} {det.meta['OBJDEC']:+6f}"
            )
            Z = np.array(
                imgwcs.all_world2pix([det.meta["OBJRA"]], [det.meta["OBJDEC"]], 1)
            ).transpose()

            if np.isnan(Z[0][0]) or np.isnan(Z[0][1]):
                object_ind = None
                logging.warning("Target transforms to Nan... :(")
            else:
                treeZ = KDTree(Z)
                object_ind, object_dist = treeZ.query_radius(
                    Y, r=idlimit, return_distance=True, count_only=False
                )

                if object_ind is not None:
                    mindist = 2 * idlimit  # this is just a big number
                    for i, dd, d in zip(object_ind, object_dist, det):
                        if len(i) > 0 and dd[0] < mindist:
                            target[imgno] = d
                            mindist = dd[0]
        #                        if target[imgno] is None:
        #                            logging.info("Target was not found")
        #                        else:
        #                            logging.info(f"Target is object id {target[imgno]['NUMBER']} at distance {mindist:.2f} px")

        # === objects identified ===

        # TODO: handle discrepancy between estimated and fitsheader filter
        #        flt = det.meta['FILTER']

        #        if options.tryflt:
        #            fltx = check_filter(nearest_ind, det)
        #        print("Filter: %s"%(fltx))
        #            flt = fltx

        # crude estimation of an image photometric zeropoint
        #        Zo, Zoe, Zn = median_zeropoint(nearest_ind, det, cat, flt)

        #        if Zn == 0:
        #            print(f"Warning: no identified stars in {det.meta['FITSFILE']}, skip image")
        #            continue

        #        if options.tryflt:
        #            print_image_line(det, flt, Zo, Zoe, target[imgno], Zn)
        #            sys.exit(0)

        # === fill up fields to be fitted ===
        #        zeropoints=zeropoints+[Zo]
        #        det.meta['IDNUM'] = Zn
        #        metadata.append(det.meta)

        # Tohle tady asi nema co delat, mozna do cat2det nebo uplne pryc?
        #        cas = astropy.time.Time(det.meta['JD'], format='jd')
        #        loc = astropy.coordinates.EarthLocation(\
        #            lat=det.meta['LATITUDE']*astropy.units.deg, \
        #            lon=det.meta['LONGITUD']*astropy.units.deg, \
        #            height=det.meta['ALTITUDE']*astropy.units.m)
        #
        #        tmpra, tmpdec = imgwcs.all_pix2world(det.meta['CTRX'], det.meta['CTRY'], 1)
        #        rd = astropy.coordinates.SkyCoord( np.float64(tmpra), np.float64(tmpdec), unit=astropy.units.deg)
        #        rdalt = rd.transform_to(astropy.coordinates.AltAz(obstime=cas, location=loc))
        #        det.meta['AIRMASS'] = airmass(np.pi/2-rdalt.alt.rad)

        det.meta["REFILTER"], det.meta["REJC"] = get_base_filter(det, options)

        # make pairs to be fitted
        det.meta["IMGNO"] = imgno
        n_matched_stars = make_pairs_to_fit(
            det, cat, nearest_ind, imgwcs, options, data
        )
        det.meta["IDNUM"] = n_matched_stars
>>>>>>> cf38f76 (Add the daemon to transient analyser)

        if n_matched_stars == 0:
            logging.warning(
                f"No matched stars in {det.meta['FITSFILE']}, skipping image"
            )
            continue

<<<<<<< HEAD
=======
        # Store metadata for later use
        #        det.meta['IMGNO'] = imgno
>>>>>>> cf38f76 (Add the daemon to transient analyser)
        metadata.append(det.meta)
        alldet.append(det)
        target.append(target_match)

        imgno += 1

    data.finalize()

    if imgno == 0:
        logging.error("No images found to work with")
        sys.exit(0)

    if len(data) == 0:
        logging.error("No objects found to work with")
        sys.exit(0)

    logging.info(
        f"Photometry will be fitted with {len(data)} objects from {imgno} files"
    )
    # tady mame hvezdy ze snimku a hvezdy z katalogu nactene a identifikovane

    start = time.time()
    ffit = perform_photometric_fitting(data, options, metadata)
    logging.info(ffit)
    logging.info(f"Photometric fit took {time.time()-start:.3f}s")

    # Update det objects if filter was changed during discovery
    filter_check_mode = getattr(options, 'filter_check', 'none')
    if filter_check_mode in ['d', 'discover']:
        final_filter = metadata[0]['PHFILTER']
        for i, det in enumerate(alldet):
            if det.meta['FILTER'] != final_filter:
                logging.info(f"Updating det object {i}: {det.meta['FILTER']} → {final_filter}")
                det.meta['FILTER'] = final_filter

    if options.reject:
        if ffit.wssrndf > options.reject:
            logging.info("rejected (too large reduced chi2)")
            out_file = open("rejected", "a+")
            out_file.write("%s %.6f -\n" % (metadata[0]["FITSFILE"], ffit.wssrndf))
            out_file.close()
            sys.exit(0)

    if options.save_model is not None:
        ffit.savemodel(options.save_model)

    # """ REFIT ASTROMETRY """
    zpntest=None # correct behaviour if astrometry is off
    if options.astrometry:
        if len(alldet) > 1:
            logging.warning(f"Astrometric fitting is not supported with multiple input files ({len(alldet)} files provided)")
            logging.warning("Astrometry disabled to prevent bogus results. Process files individually for astrometric fitting.")
            print(f"WARNING: Astrometry disabled - {len(alldet)} input files detected")
            print("WARNING: Astrometric fitting only works with single images. Process files individually.")
        else:
            start = time.time()
            zpntest = refit_astrometry(det, data, options)
            logging.info(f"Astrometric fit took {time.time()-start:.3f}s")
        if zpntest is not None:
            # Update FITS file with new WCS
            start = time.time()
            fitsbase = os.path.splitext(arg)[0]
            newfits = fitsbase + "t.fits"
            if os.path.isfile(newfits):
                logging.info(f"Will overwrite {newfits}")
                os.unlink(newfits)
            os.system(f"cp {fitsbase}.fits {newfits}")
            for key in ['FIELD', 'PIXEL', 'FWHM']:
                astropy.io.fits.setval(newfits, key, 0, value=det.meta[key])
            zpntest.write(newfits)
            zpntest.write(det.meta)
            imgwcs = astropy.wcs.WCS(zpntest.wcs())
            logging.info(f"Saving a new fits with WCS took {time.time()-start:.3f}s")
    # ASTROMETRY END

    if options.stars:
        start = time.time()
        write_stars_file(data, ffit, imgwcs)
        logging.info(f"Saving the stars file took {time.time()-start:.3f}s")

    if options.plot:
        start = time.time()
        base_filename = os.path.splitext(det.meta['detf'])[0]
        create_residual_plots(data, base_filename, ffit, zpntest, 'photometry')
        create_correction_volume_plots(data, base_filename, ffit)
        logging.info(f"Generating plots took {time.time()-start:.3f}s")

    if not options.remove_spatial:
        write_results(data, ffit, options, alldet, target, zpntest)


# this way, the variables local to main() are not globally available, avoiding some programming errors
if __name__ == "__main__":
    main()
