#!/usr/bin/python3

import logging
import time

import hashlib
from typing import Any, Dict, Optional, List, Tuple, cast

import astropy.table
import astropy.wcs
import numpy as np
from sklearn.neighbors import KDTree
from catalog import Catalog, CatalogOptimizationCache
import astropy

class CatTransients(Catalog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def transform_to_instrumental(
        self, det: astropy.table.Table, wcs: astropy.wcs.WCS
    ) -> Optional[astropy.table.Table]:
        """Transform catalog to instrumental system."""
        try:
            cat_out = super().copy()
            target_filter = det.meta.get("REFILTER")
            if not target_filter:
                raise ValueError(
                    "No target filter (REFILTER) specified in detection metadata"
                )

            # Create color selector and get colors
            selector = ColorSelector(self.filters)
            colors, color_descriptions = selector.prepare_color_terms(
                self, target_filter
            )

            cat_out = self.copy()
            cat_out.meta["color_terms"] = color_descriptions
            cat_out.meta["target_filter"] = target_filter

            try:
                cat_x, cat_y = wcs.all_world2pix(self["radeg"], self["decdeg"], 1)
            except Exception as e:
                raise ValueError(f"Coordinate transformation failed: {str(e)}")

            if "RESPONSE" not in det.meta:
                raise ValueError("No RESPONSE model in detection metadata")

            try:
                import fotfit

                ffit = fotfit.FotFit()
                ffit.from_oneline(det.meta["RESPONSE"])
            except Exception as e:
                raise ValueError(f"Failed to load photometric model: {str(e)}")

            filter_info = self.filters[target_filter]
            base_mag = self[filter_info.name]

            model_input = (
                base_mag,
                det.meta["AIRMASS"],
                (cat_x - det.meta["CTRX"]) / 1024,
                (cat_y - det.meta["CTRY"]) / 1024,
                colors[0],
                colors[1],
                colors[2],
                colors[3],
                det.meta["IMGNO"],
                np.zeros_like(base_mag),
                np.ones_like(base_mag),
            )

            cat_out["mag_instrument"] = ffit.model(ffit.fixvalues, model_input)

            if filter_info.error_name and filter_info.error_name in self.columns:
                cat_out["mag_instrument_err"] = np.sqrt(
                    self[filter_info.error_name] ** 2 + 0.01 ** 2
                )
            else:
                cat_out["mag_instrument_err"] = np.full_like(base_mag, 0.03)

            if "catalog_props" in self.meta:
                cat_out.meta["catalog_props"] = self.meta["catalog_props"].copy()
            
            cat_out.meta["transform_info"] = {
                "source_catalog": self.catalog_name,
                "source_filter": filter_info.name,
                "target_filter": target_filter,
                "color_terms": color_descriptions,
                "airmass": float(det.meta["AIRMASS"]),
                "model": det.meta["RESPONSE"],
            }

            return cat_out

        except Exception as e:
            raise ValueError(f"Transformation failed: {str(e)}")

    def get_transient_candidates(
        self, det: astropy.table.Table, idlimit: float = 5.0
    ) -> astropy.table.Table:
        """Identify transient candidates using optimized cached methods."""
        # Deprecation warning for legacy method
        import warnings
        warnings.warn(
            "get_transient_candidates is deprecated. Use get_transient_candidates_optimized for better performance.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Delegate to optimized version
        return self.get_transient_candidates_optimized(det, idlimit=idlimit)
    

    def compute_magnitude_difference(
        self, det_without_transients: astropy.table.Table, filter: str
    ) -> astropy.table.Table:
        """Compute magnitude differences between detections and catalog sources."""
        if filter not in self.filters:
            raise ValueError(f"Filter '{filter}' not available in the catalog.")
        try:

            self._validate_detection_table(det_without_transients)

            cat_xy = self._transform_catalog_to_pixel(det_without_transients)

            det_xy = np.array(
                [det_without_transients["X_IMAGE"], det_without_transients["Y_IMAGE"]]
            ).T

            tree = KDTree(cat_xy)

            dist, idx = tree.query(det_xy, k=1)
            
            if filter in self.filters:
                cat_mag = (self[filter][idx]).flatten()
                det_mag = np.array(det_without_transients["MAG_CALIB"])
                det_without_transients["mag_diff"] = det_mag - cat_mag
                try:
                    cat_err = self[self.filters[filter].error_name][idx]
                    det_without_transients["mag_diff_err"] = np.sqrt(
                        cat_err ** 2
                        + np.array(det_without_transients["MAGERR_CALIB"]) ** 2
                    )
                except Exception:
                    pass

            det_without_transients.meta.update(
                {"reference_catalog": self.catalog_name, "matching_time": time.time()}
            )

            return det_without_transients

        except Exception as e:
            raise ValueError(
                f"Magnitude difference computation failed: {str(e)}"
            ) from e

    def _validate_detection_table(self, det: astropy.table.Table) -> None:
        """Validate detection table has required columns and metadata."""
        required_columns = {"X_IMAGE", "Y_IMAGE"}
        if missing_columns := required_columns - set(det.colnames):
            raise ValueError(
                f"Detection table missing required columns: {missing_columns}"
            )

        # Check for image dimensions using fallback chain
        width_keys = ['NAXIS1', 'IMGAXIS1', 'IMAGEW']
        height_keys = ['NAXIS2', 'IMGAXIS2', 'IMAGEH']
        
        has_width = any(key in det.meta for key in width_keys)
        has_height = any(key in det.meta for key in height_keys)
        
        if not has_width:
            raise ValueError(f"Detection table missing width metadata: tried {width_keys}")
        if not has_height:
            raise ValueError(f"Detection table missing height metadata: tried {height_keys}")

    def _transform_catalog_to_pixel(self, det: astropy.table.Table) -> np.ndarray:
        """Transform catalog coordinates to pixel coordinates without mutating input."""
        try:
            # Copy header to avoid mutating det.meta
            header = dict(det.meta)
            header['CTYPE1'] = 'RA---TAN'
            header['CTYPE2'] = 'DEC--TAN'
            
            # Remove potentially problematic keys
            keys_to_remove = ['CTYPE1T', 'CTYPE2T', 'CRVAL1T', 'CRVAL2T', 
                             'CDELT1T', 'CDELT2T', 'CROTA2T']
            for key in keys_to_remove:
                header.pop(key, None)
            
            # Remove distortion keys that might cause issues
            distortion_patterns = ['PV', 'A_', 'B_', 'AP_', 'BP_']
            keys_to_remove = [k for k in header.keys() 
                             if any(pattern in k for pattern in distortion_patterns)]
            for key in keys_to_remove:
                header.pop(key, None)

            imgwcs = astropy.wcs.WCS(header)
            cat_x, cat_y = imgwcs.all_world2pix(self["radeg"], self["decdeg"], 1)

            return np.column_stack([cat_x, cat_y])
        except Exception as e:
            raise ValueError(f"Coordinate transformation failed: {str(e)}") from e

    def match_with_external_catalog(
        self, other_cat: "Catalog", max_separation: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Match sources with another catalog using sky coordinates."""
        import astropy.units as u
        from astropy.coordinates import SkyCoord

        cat1_coords = SkyCoord(ra=self["radeg"] * u.deg, dec=self["decdeg"] * u.deg)
        cat2_coords = SkyCoord(
            ra=other_cat["radeg"] * u.deg, dec=other_cat["decdeg"] * u.deg
        )

        idx1, idx2, sep, _ = cat1_coords.search_around_sky(
            cat2_coords, max_separation * u.arcsec
        )

        return idx1, idx2
    def precompute_photometric_data(self, bands: Optional[List[str]] = None, 
                                  force_recompute: bool = False) -> CatalogOptimizationCache:
        """
        Precompute photometric data for fast access in transient detection.
        
        Args:
            bands: List of photometric bands to precompute (default: standard 5-band)
            force_recompute: Force recomputation even if cache exists
            
        Returns:
            CatalogCache object with precomputed data
        """
        if self._photometric_cache is not None and not force_recompute:
            return self._photometric_cache
        
        if bands is None:
            bands = ["Sloan_g", "Sloan_r", "Sloan_i", "Sloan_z", "J"]
        
        logging.info(f"Precomputing photometric data for {len(self)} stars...")
        
        # Extract coordinates
        coordinates = np.column_stack([self["radeg"], self["decdeg"]])
        
        # Precompute photometric data
        n_stars = len(self)
        magnitudes = np.full((n_stars, len(bands)), np.nan)
        colors = np.full((n_stars, len(bands)-1), np.nan)
        
        # Extract magnitudes efficiently
        for i, band in enumerate(bands):
            if band in self.columns:
                mag_col = self[band]
                # Handle masked arrays and invalid values
                if hasattr(mag_col, 'mask'):
                    valid = ~mag_col.mask & np.isfinite(mag_col.data) & (mag_col.data < 99)
                    magnitudes[valid, i] = mag_col.data[valid]
                else:
                    valid = np.isfinite(mag_col) & (mag_col < 99)
                    magnitudes[valid, i] = mag_col[valid]
        
        # Calculate colors for stars with sufficient photometry
        valid_stars = np.sum(~np.isnan(magnitudes), axis=1) >= 2
        n_valid = np.sum(valid_stars)
        
        logging.info(f"Processing colors for {n_valid} stars with sufficient photometry...")
        
        for i in np.where(valid_stars)[0]:
            star_mags = magnitudes[i].copy()
            
            # Fill missing bands using typical colors
            filled_mags = self.fill_missing_photometry(star_mags)
            
            if filled_mags is not None:
                magnitudes[i] = filled_mags
                # Calculate standard colors
                if len(filled_mags) >= 5:
                    colors[i] = [
                        filled_mags[0] - filled_mags[1],  # g-r
                        filled_mags[1] - filled_mags[2],  # r-i
                        filled_mags[2] - filled_mags[3],  # i-z
                        filled_mags[3] - filled_mags[4],  # z-J
                    ]
        
        # Create cache object
        self._photometric_cache = CatalogOptimizationCache(
            coordinates=coordinates,
            pixel_coordinates={},
            magnitudes=magnitudes,
            colors=colors,
            valid_stars=valid_stars,
            kdtrees={}
        )
        
        logging.info(f"Photometric precomputation complete: {n_valid} valid stars")
        return self._photometric_cache
    
    @staticmethod
    def fill_missing_photometry(mags: np.ndarray, 
                               typical_colors: Optional[List[float]] = None) -> Optional[np.ndarray]:
        """
        Fill missing photometric bands using typical stellar colors.
        
        Args:
            mags: Array of magnitudes with possible NaN values
            typical_colors: Typical color indices (default: [0.6, 0.3, 0.2, 0.8] for g-r, r-i, i-z, z-J)
            
        Returns:
            Filled magnitude array or None if insufficient data
        """
        if typical_colors is None:
            typical_colors = [0.6, 0.3, 0.2, 0.8]  # g-r, r-i, i-z, z-J
        
        if np.sum(~np.isnan(mags)) < 2:
            return None
        
        filled = mags.copy()
        
        # Forward fill using typical colors
        for i in range(len(typical_colors)):
            if i+1 < len(filled):
                if not np.isnan(filled[i]) and np.isnan(filled[i+1]):
                    filled[i+1] = filled[i] + typical_colors[i]
        
        # Backward fill using typical colors
        for i in range(len(typical_colors)-1, -1, -1):
            if i+1 < len(filled):
                if not np.isnan(filled[i+1]) and np.isnan(filled[i]):
                    filled[i] = filled[i+1] - typical_colors[i]
        
        # Final fill with last valid value
        last_valid = None
        for i in range(len(filled)):
            if not np.isnan(filled[i]):
                last_valid = filled[i]
            elif last_valid is not None:
                filled[i] = last_valid
        
        return filled if not np.any(np.isnan(filled)) else None
    
    def get_pixel_coordinates_cached(self, detections: astropy.table.Table, 
                                   image_id: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Get pixel coordinates with caching for reuse across multiple operations.
        
        Args:
            detections: Detection table with WCS information
            image_id: Unique identifier for caching (auto-generated if None)
            
        Returns:
            Array of pixel coordinates or None if transformation fails
        """
        if not self._cache_enabled:
            return self._transform_catalog_to_pixel(detections)
        
        if image_id is None:
            image_id = self._generate_image_id(detections)
        
        # Check cache first
        if self._photometric_cache and image_id in self._photometric_cache.pixel_coordinates:
            logging.debug(f"Using cached pixel coordinates for {image_id}")
            return self._photometric_cache.pixel_coordinates[image_id]
        
        # Compute and cache with timing
        try:
            import time
            start_time = time.time()
            
            coords = self._transform_catalog_to_pixel(detections)
            
            elapsed = time.time() - start_time
            logging.debug(f"WCS transform completed in {elapsed:.3f}s for {len(self)} stars -> {image_id}")
            
            if self._photometric_cache is not None:
                self._photometric_cache.pixel_coordinates[image_id] = coords
            else:
                self._coordinate_cache[image_id] = coords
            
            return coords
        except Exception as e:
            logging.debug(f"WCS transform failed: {e}")
            return None
    
    def build_spatial_index(self, coordinates: np.ndarray, 
                          index_id: Optional[str] = None) -> KDTree:
        """
        Build and cache KDTree for spatial queries, guaranteed caching by image ID.
        
        Args:
            coordinates: Coordinate array for tree building
            index_id: Unique identifier for caching (auto-generated if None)
            
        Returns:
            KDTree object
        """
        # Generate index_id if not provided to guarantee caching
        if index_id is None and self._cache_enabled:
            # Create a hash from the coordinates for stable ID
            coord_hash = hashlib.md5(coordinates.tobytes()).hexdigest()[:12]
            index_id = f"kdtree_{coord_hash}"
        
        # If caching disabled or no ID, build fresh tree with timing
        if not self._cache_enabled or index_id is None:
            import time
            start_time = time.time()
            
            tree = KDTree(coordinates)
            
            elapsed = time.time() - start_time
            logging.debug(f"Built uncached KDTree in {elapsed:.3f}s for {len(coordinates)} points")
            return tree
        
        # Check cache - try photometric cache first, then fallback cache
        cache_dict = (self._photometric_cache.kdtrees if self._photometric_cache 
                     else self._kdtree_cache)
        
        if index_id not in cache_dict:
            import time
            start_time = time.time()
            
            cache_dict[index_id] = KDTree(coordinates)
            
            elapsed = time.time() - start_time
            logging.debug(f"Built and cached KDTree in {elapsed:.3f}s for {len(coordinates)} points -> {index_id}")
        else:
            logging.debug(f"Using cached KDTree for {index_id} ({len(coordinates)} points)")
        
        return cache_dict[index_id]
    
    def compute_local_statistics(self, positions: np.ndarray, radius: float,
                               filter_pattern: Optional[str] = None,
                               image_id: Optional[str] = None) -> Dict[str, List]:
        """
        Compute local statistics around detection positions using cached catalog pixel coordinates.
        
        Args:
            positions: Detection pixel coordinates (X_IMAGE, Y_IMAGE) - used as query points
            radius: Search radius in pixels around each detection position
            filter_pattern: Filter name for magnitude statistics
            image_id: Image identifier for cached coordinates (prevents cross-image mismatches)
            
        Returns:
            Dictionary with statistics for each detection position, safe defaults on cache miss
        """
        n_positions = len(positions)
        
        # Safe defaults for all outputs
        default_results = {
            'nearby_sources': [0] * n_positions,
            'source_density': [0.0] * n_positions,
            'nearest_source_dist': [np.inf] * n_positions
        }
        
        if filter_pattern:
            default_results[f'mean_mag_{filter_pattern}'] = [np.nan] * n_positions
            default_results[f'std_mag_{filter_pattern}'] = [np.nan] * n_positions
        
        if n_positions == 0:
            return default_results

        # ALWAYS use cached catalog pixel coordinates - this is the key requirement
        cat_coords = None
        
        try:
            if (self._photometric_cache and 
                hasattr(self._photometric_cache, 'pixel_coordinates') and
                len(self._photometric_cache.pixel_coordinates) > 0):
                
                if image_id and image_id in self._photometric_cache.pixel_coordinates:
                    # Use specific image coordinates
                    cat_coords = self._photometric_cache.pixel_coordinates[image_id]
                else:
                    # Use most recent cached coordinates
                    coord_key = list(self._photometric_cache.pixel_coordinates.keys())[0]
                    cat_coords = self._photometric_cache.pixel_coordinates[coord_key]
                
        except Exception as e:
            logging.debug(f"Could not access cached catalog pixel coordinates: {e}")
        
        # Return defaults if no cached coordinates available (avoid exceptions)
        if cat_coords is None or len(cat_coords) == 0:
            logging.debug("No cached catalog pixel coordinates available, returning defaults")
            return default_results
        
        try:
            # Build spatial index using cached coordinates
            tree_id = f"stats_{image_id}" if image_id else None
            tree = self.build_spatial_index(cat_coords, tree_id)
            
            # Find neighbors for each position
            neighbors = tree.query_radius(positions, r=radius)
            
            # Calculate statistics
            nearby_sources = [len(n) for n in neighbors]
            source_density = [count / (np.pi * radius ** 2) for count in nearby_sources]
            
            # Get nearest neighbor distances
            distances, _ = tree.query(positions, k=1)
            nearest_distances = distances.flatten().tolist()
            
            results = {
                'nearby_sources': nearby_sources,
                'source_density': source_density,
                'nearest_source_dist': nearest_distances
            }
            
            # Add filter-specific statistics using precomputed data
            if filter_pattern and self._photometric_cache is not None:
                try:
                    filter_stats = self._compute_filter_statistics(neighbors, filter_pattern)
                    results.update(filter_stats)
                except Exception as e:
                    logging.debug(f"Could not compute filter statistics: {e}")
                    # Use defaults from above
                    results[f'mean_mag_{filter_pattern}'] = [np.nan] * n_positions
                    results[f'std_mag_{filter_pattern}'] = [np.nan] * n_positions
            
            return results
            
        except Exception as e:
            logging.debug(f"Statistics computation failed: {e}, returning defaults")
            return default_results
    
    def _compute_filter_statistics(self, neighbors: List[np.ndarray], 
                                 filter_pattern: str) -> Dict[str, List]:
        """Compute filter-specific statistics using precomputed data."""
        
        # Map filter pattern to band index
        band_mapping = {
            'g': 0, 'r': 1, 'i': 2, 'z': 3, 'j': 4,
            'sloan_g': 0, 'sloan_r': 1, 'sloan_i': 2, 'sloan_z': 3
        }
        
        band_idx = band_mapping.get(filter_pattern.lower(), 1)  # Default to r-band
        
        mean_mags = []
        std_mags = []
        
        for neighbor_indices in neighbors:
            if len(neighbor_indices) > 0 and self._photometric_cache is not None:
                neighbor_mags = self._photometric_cache.magnitudes[neighbor_indices, band_idx]
                valid_mags = neighbor_mags[~np.isnan(neighbor_mags)]
                
                if len(valid_mags) > 0:
                    mean_mags.append(np.mean(valid_mags))
                    std_mags.append(np.std(valid_mags) if len(valid_mags) > 1 else 0.0)
                else:
                    mean_mags.append(np.nan)
                    std_mags.append(np.nan)
            else:
                mean_mags.append(np.nan)
                std_mags.append(np.nan)
        
        return {
            f'mean_mag_{filter_pattern}': mean_mags,
            f'std_mag_{filter_pattern}': std_mags
        }
    
    def _compute_adaptive_radii(self, detections: astropy.table.Table, 
                               nsigma: float = 3.0, 
                               idlimit_min_px: float = 1.0,
                               idlimit_max_px: float = 8.0, 
                               use_astvar: bool = True) -> np.ndarray:
        """
        Compute per-detection adaptive identification radii based on centroid uncertainties.
        
        Formula: r_i = nsigma * sqrt(ERRX2_IMAGE + ERRY2_IMAGE) * sqrt(ASTVAR)
        
        Args:
            detections: Detection table
            nsigma: Number of sigma for radius computation
            idlimit_min_px: Minimum allowed radius in pixels
            idlimit_max_px: Maximum allowed radius in pixels  
            use_astvar: Whether to use ASTVAR scaling factor
            
        Returns:
            Array of radii in pixels, or empty array if computation fails
        """
        if len(detections) == 0:
            return np.array([])
        
        try:
            # Try primary method: ERRX2_IMAGE + ERRY2_IMAGE
            if 'ERRX2_IMAGE' in detections.colnames and 'ERRY2_IMAGE' in detections.colnames:
                errx2 = detections['ERRX2_IMAGE'].data
                erry2 = detections['ERRY2_IMAGE'].data
                
                # Check for valid error values
                valid_errors = np.isfinite(errx2) & np.isfinite(erry2) & (errx2 >= 0) & (erry2 >= 0)
                
                if np.any(valid_errors):
                    # Compute positional uncertainty
                    pos_err = np.sqrt(errx2 + erry2)
                    
                    # Apply ASTVAR scaling if requested
                    if use_astvar:
                        astvar = detections.meta.get('ASTVAR', 1.0)
                        if not np.isfinite(astvar) or astvar <= 0:
                            astvar = 1.0
                        pos_err *= np.sqrt(astvar)
                    
                    # Compute adaptive radii
                    radii = nsigma * pos_err
                    
                    # Handle invalid values
                    radii = np.where(valid_errors & np.isfinite(radii), radii, np.nan)
                    
                    logging.debug(f"Computed adaptive radii from ERRX2/ERRY2_IMAGE: "
                                f"{np.sum(np.isfinite(radii))} valid of {len(radii)} detections")
                    
                    # If we have some valid radii, use them
                    if np.sum(np.isfinite(radii)) > 0:
                        # Clamp to limits
                        radii = np.clip(radii, idlimit_min_px, idlimit_max_px)
                        return radii
            
            # Fallback 1: Compute from SNR
            logging.debug("Falling back to SNR-based radius computation")
            snr_values = None
            
            if 'SNR' in detections.colnames:
                snr_values = detections['SNR'].data
            elif 'FLUX_ISO' in detections.colnames and 'FLUXERR_ISO' in detections.colnames:
                flux = detections['FLUX_ISO'].data
                fluxerr = detections['FLUXERR_ISO'].data
                valid_flux = (fluxerr > 0) & np.isfinite(flux) & np.isfinite(fluxerr)
                snr_values = np.where(valid_flux, flux / fluxerr, np.nan)
                logging.debug("Computed SNR from FLUX_ISO/FLUXERR_ISO")
            
            if snr_values is not None:
                valid_snr = np.isfinite(snr_values) & (snr_values > 0)
                
                if np.any(valid_snr):
                    # Get FWHM for PSF sigma computation
                    fwhm_values = None
                    if 'FWHM_IMAGE' in detections.colnames:
                        fwhm_values = detections['FWHM_IMAGE'].data
                    else:
                        # Try to get from metadata
                        fwhm_meta = detections.meta.get('FWHM', 1.2)
                        fwhm_values = np.full(len(detections), fwhm_meta)
                        logging.debug(f"Using FWHM from metadata: {fwhm_meta}")
                    
                    # Compute PSF sigma and positional uncertainty
                    eps = 1e-6
                    effective_snr = np.maximum(snr_values, eps)
                    fwhm_eff = np.where(np.isfinite(fwhm_values) & (fwhm_values > 0), 
                                       fwhm_values, 1.2)
                    
                    # PSF sigma = FWHM / 2.35, positional error ~ sigma / SNR
                    psf_sigma = fwhm_eff / 2.35
                    pos_err = psf_sigma / effective_snr
                    
                    # Apply ASTVAR scaling if requested
                    if use_astvar:
                        astvar = detections.meta.get('ASTVAR', 1.0)
                        if not np.isfinite(astvar) or astvar <= 0:
                            astvar = 1.0
                        pos_err *= np.sqrt(astvar)
                    
                    # Compute adaptive radii
                    radii = nsigma * pos_err
                    radii = np.where(valid_snr & np.isfinite(radii), radii, np.nan)
                    
                    logging.debug(f"Computed adaptive radii from SNR/FWHM: "
                                f"{np.sum(np.isfinite(radii))} valid of {len(radii)} detections")
                    
                    if np.sum(np.isfinite(radii)) > 0:
                        # Clamp to limits
                        radii = np.clip(radii, idlimit_min_px, idlimit_max_px)
                        return radii
            
            # All fallbacks failed
            logging.warning("All adaptive radius computation methods failed, no valid radii computed")
            return np.array([])
            
        except Exception as e:
            logging.error(f"Error in adaptive radius computation: {e}")
            return np.array([])
    
    def get_transient_candidates_optimized(self, detections: astropy.table.Table,
                                         idlimit: float = 5.0,
                                         mag_change_threshold: float = 1.0,
                                         siglim: float = 5.0,
                                         frame: float = 10.0,
                                         adaptive_radii: Optional[np.ndarray] = None) -> astropy.table.Table:
        """
        Optimized transient candidate detection using precomputed catalog data.
        
        Args:
            detections: Detection table
            idlimit: Default matching radius in pixels (used when adaptive_radii is None)
            mag_change_threshold: Magnitude change threshold for variability detection
            siglim: Sigma threshold for significance
            frame: Frame edge exclusion in pixels
            adaptive_radii: Optional array of per-detection radii in pixels
            
        Returns:
            Table of transient candidates with type and magnitude difference info
        """
        if len(detections) == 0:
            return astropy.table.Table()
        
        # Ensure we have precomputed data - this is now the default path
        if self._photometric_cache is None:
            logging.debug("Auto-precomputing catalog data for optimized detection...")
            self.precompute_photometric_data()
        
        # Get pixel coordinates (cached) - default path
        image_id = self._generate_image_id(detections)
        cat_xy = self.get_pixel_coordinates_cached(detections, image_id)
        
        # Build spatial index (cached) - default path
        tree = self.build_spatial_index(cat_xy, image_id)
        
        # Get detection coordinates
        det_xy = np.column_stack([detections["X_IMAGE"], detections["Y_IMAGE"]])
        
        # Check for adaptive identification configuration in metadata or use provided radii
        adaptive_enabled = False
        r_i = adaptive_radii
        
        # Read config from detections.meta if adaptive_radii not provided
        if adaptive_radii is None:
            adaptive_enabled = detections.meta.get('adaptive_idlimit_enabled', False)
            if adaptive_enabled:
                nsigma = detections.meta.get('adaptive_nsigma', 3.0)
                idlimit_min_px = detections.meta.get('idlimit_min_px', 1.0)
                idlimit_max_px = detections.meta.get('idlimit_max_px', 8.0)
                use_astvar = detections.meta.get('use_astvar', True)
                
                r_i = self._compute_adaptive_radii(detections, nsigma, idlimit_min_px, idlimit_max_px, use_astvar)
                
                # Check if computation failed
                if len(r_i) == 0 or np.all(~np.isfinite(r_i)):
                    logging.warning("Adaptive radius computation failed, falling back to fixed idlimit")
                    adaptive_enabled = False
                    r_i = None
        else:
            adaptive_enabled = True
        
        # Adaptive matching with percentile query + post-filter
        if adaptive_enabled and r_i is not None:
            if len(r_i) != len(detections):
                raise ValueError(f"adaptive_radii length ({len(r_i)}) must match detections length ({len(detections)})")
            
            # Validate and clean adaptive radii
            valid_radii = np.isfinite(r_i) & (r_i > 0)
            if not np.any(valid_radii):
                logging.warning("No valid adaptive radii found, falling back to fixed idlimit")
                # Fall through to fixed radius path
                adaptive_enabled = False
            else:
                # Replace invalid radii with default
                r_i = np.where(valid_radii, r_i, idlimit)
                
                # Get percentile for query radius
                adaptive_percentile = detections.meta.get('adaptive_percentile', 95.0)
                r_query = np.percentile(r_i, adaptive_percentile)
                
                # Log statistics
                logging.info(f"Adaptive matching: {len(detections)} detections, "
                           f"r_i range {np.min(r_i):.2f}-{np.max(r_i):.2f}px "
                           f"(median: {np.median(r_i):.2f}px)")
                logging.debug(f"Using r_query = {r_query:.2f}px ({adaptive_percentile}th percentile)")
                
                # Single KDTree query with return_distance=True
                matches_indices, distances = tree.query_radius(det_xy, r=r_query, return_distance=True)
                
                # Post-filter: keep only neighbors within per-detection radius
                matches_list = []
                total_matches_before = sum(len(matches) for matches in matches_indices)
                total_matches_after = 0
                
                for i, (matches, dists) in enumerate(zip(matches_indices, distances)):
                    # Filter by per-detection radius
                    valid_matches = dists <= r_i[i]
                    filtered_matches = matches[valid_matches]
                    matches_list.append(filtered_matches)
                    total_matches_after += len(filtered_matches)
                
                # Log filtering statistics
                filtered_out = total_matches_before - total_matches_after
                logging.debug(f"r_query found {total_matches_before} neighbor pairs, "
                            f"filtered out {filtered_out}, kept {total_matches_after}")
        
        # Fixed radius matching (fallback or when adaptive disabled)
        if not adaptive_enabled:
            logging.info(f"Using fixed matching: {len(detections)} detections, radius {idlimit:.2f}px")
            matches_list = tree.query_radius(det_xy, r=idlimit)
            total_matches = sum(len(matches) for matches in matches_list)
            logging.debug(f"Fixed matching found {total_matches} total catalog matches")
        
        # Process detections with vectorized quality filtering
        return self._process_detections_for_candidates(
            detections, matches_list, mag_change_threshold, siglim, frame
        )
    
    def _process_detections_for_candidates(self, detections: astropy.table.Table,
                                         matches_list: List[np.ndarray],
                                         mag_change_threshold: float,
                                         siglim: float, frame: float) -> astropy.table.Table:
        """Process detections to find candidates using precomputed data."""
        
        # Vectorized quality filtering
        det_x = detections["X_IMAGE"].data
        det_y = detections["Y_IMAGE"].data
        det_mags = detections["MAG_CALIB"].data
        det_mag_errs = detections["MAGERR_CALIB"].data
        
        # Get image dimensions with safe fallback chain
        img_w = detections.meta.get('NAXIS1', 
                detections.meta.get('IMGAXIS1', 
                detections.meta.get('IMAGEW', np.max(det_x) + 100)))
        img_h = detections.meta.get('NAXIS2', 
                detections.meta.get('IMGAXIS2', 
                detections.meta.get('IMAGEH', np.max(det_y) + 100)))
        
        # Vectorized exclusion masks
        edge_mask = ((det_x < frame) | (det_y < frame) | 
                    (det_x > img_w - frame) | (det_y > img_h - frame))
        quality_mask = det_mag_errs >= (1.091 / siglim)
        exclude_mask = edge_mask | quality_mask
        
        # Process good detections
        candidates = []
        candidate_types = []
        mag_differences = []
        
        response_model = detections.meta.get("RESPONSE", "P0=25.0")
        
        for i, matches in enumerate(matches_list):
            if exclude_mask[i]:
                continue
            
            if len(matches) == 0:
                # No matches - new transient
                candidates.append(i)
                candidate_types.append('new')
                mag_differences.append(0.0)
                continue
            
            # Check for magnitude changes using precomputed data
            is_cand, cand_type, mag_diff = self._check_magnitude_changes_cached(
                matches, det_mags[i], det_mag_errs[i], response_model, 
                mag_change_threshold, siglim
            )
            
            if is_cand:
                candidates.append(i)
                candidate_types.append(cand_type)
                mag_differences.append(mag_diff)
        
        # Create result table
        if candidates:
            result = detections[candidates].copy()
            result['candidate_type'] = candidate_types
            result['magnitude_difference'] = mag_differences
            return result
        else:
            return astropy.table.Table()
    
    def _check_magnitude_changes_cached(self, matches: np.ndarray, det_mag: float,
                                      det_mag_err: float, response_model: str,
                                      mag_change_threshold: float, siglim: float) -> Tuple[bool, str, float]:
        """Check for magnitude changes using precomputed catalog data."""
        
        # Get catalog systematic floor based on catalog name
        catalog_name = self.catalog_name.lower()
        if 'gaia' in catalog_name:
            cat_sys_floor = 0.01
        elif 'panstarrs' in catalog_name:
            cat_sys_floor = 0.02
        elif 'atlas' in catalog_name:
            cat_sys_floor = 0.03
        elif 'usno' in catalog_name:
            cat_sys_floor = 0.10
        else:
            cat_sys_floor = 0.02  # Default
        
        det_sys_floor = 0.01
        
        significant_changes = []
        
        for match_idx in matches:
            if not self._photometric_cache.valid_stars[match_idx]:
                continue
            
            try:
                # Use precomputed values
                r_mag = self._photometric_cache.magnitudes[match_idx, 1]  # r-band
                colors = self._photometric_cache.colors[match_idx]        # Precomputed colors
                
                if np.isnan(r_mag) or np.any(np.isnan(colors)):
                    continue
                
                # Apply color model (requires import of simple_color_model)
                try:
                    from transients import simple_color_model
                    cat_mag = simple_color_model(
                        response_model, 
                        (r_mag, colors[0], colors[1], colors[2], colors[3])
                    )
                except ImportError:
                    # Fallback: use r-band magnitude directly if color model not available
                    cat_mag = r_mag
                
                # Compute conservative total magnitude uncertainty
                sigma_total = np.sqrt(det_mag_err**2 + det_sys_floor**2 + cat_sys_floor**2)
                
                # Check significance using total uncertainty
                mag_diff = det_mag - cat_mag
                mag_diff_sigma = abs(mag_diff) / sigma_total
                
                if abs(mag_diff) >= mag_change_threshold and mag_diff_sigma > siglim:
                    change_type = 'brightening' if mag_diff < 0 else 'fading'
                    significant_changes.append((mag_diff, change_type))
                elif abs(mag_diff) <= siglim * sigma_total:
                    # Within scatter - not a candidate
                    return False, 'none', mag_diff
                    
            except Exception:
                continue
        
        if significant_changes:
            # Return most significant change
            most_significant = max(significant_changes, key=lambda x: abs(x[0]))
            return True, most_significant[1], most_significant[0]
        
        return True, 'new', np.nan  # Couldn't match properly
    
    def _generate_image_id(self, detections: astropy.table.Table) -> str:
        """Generate stable unique identifier for image caching based on WCS-defining keys only."""
        import hashlib
        
        meta = detections.meta
        
        # Extract only WCS-defining keys for stable caching
        wcs_keys = {}
        wcs_defining_patterns = [
            'CRVAL', 'CRPIX', 'CD', 'CDELT', 'CROTA', 'NAXIS'
        ]
        
        for key, value in meta.items():
            if any(pattern in key for pattern in wcs_defining_patterns):
                # Round floating point values for stability
                if isinstance(value, (float, np.floating)):
                    wcs_keys[key] = round(float(value), 8)
                else:
                    wcs_keys[key] = value
        
        # Create hash from sorted WCS keys
        wcs_string = str(sorted(wcs_keys.items()))
        image_hash = hashlib.md5(wcs_string.encode()).hexdigest()[:12]
        
        return f"img_{image_hash}"
    
    def clear_cache(self) -> None:
        """Clear all cached data to free memory."""
        self._photometric_cache = None
        self._coordinate_cache.clear()
        self._kdtree_cache.clear()
        logging.info("Catalog cache cleared")
    
    def get_runtime_cache_info(self) -> Dict[str, Any]:
        """Get information about current runtime cache state."""
        info = {
            'photometric_cache_exists': self._photometric_cache is not None,
            'coordinate_cache_size': len(self._coordinate_cache),
            'kdtree_cache_size': len(self._kdtree_cache),
            'cache_enabled': self._cache_enabled
        }
        
        if self._photometric_cache:
            info.update({
                'n_valid_stars': np.sum(self._photometric_cache.valid_stars),
                'n_cached_coordinates': len(self._photometric_cache.pixel_coordinates),
                'n_cached_kdtrees': len(self._photometric_cache.kdtrees)
            })
        
        return info
    
    def enable_cache(self, enabled: bool = True) -> None:
        """Enable or disable caching."""
        self._cache_enabled = enabled
        if not enabled:
            self.clear_cache()

def add_catalog_argument(parser: Any) -> None:
    """Add catalog selection argument to argument parser."""
    parser.add_argument(
        "--catalog",
        choices=Catalog.KNOWN_CATALOGS.keys(),
        default="ATLAS",
        help="Catalog to use for photometric reference",
    )

