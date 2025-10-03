import os
from pathlib import Path
import warnings
import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from itertools import chain
from astropy.table import Table, vstack
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import SkyCoord
from sklearn.neighbors import KDTree
from typing import List, Dict, Optional

from catalog import Catalog, QueryParams
from dataclasses import dataclass
from collections import defaultdict


class UnionFind:
    """Union-Find (Disjoint Set) data structure for efficient connected components."""
    
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.n_components = n
    
    def find(self, x: int) -> int:
        """Find with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: int, y: int) -> bool:
        """Union by rank. Returns True if components were merged."""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False
        
        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        
        self.n_components -= 1
        return True
    
    def get_components(self) -> Dict[int, List[int]]:
        """Get all connected components as dict of root -> [members]."""
        components = defaultdict(list)
        for i in range(len(self.parent)):
            root = self.find(i)
            components[root].append(i)
        return dict(components)



def combine_results(
    transients: Dict[str, Table], min_catalogs: int = 1, min_quality: float = 0.5,
    use_sky_coords: bool = True, position_match_radius_arcsec: float = 1.0,
    config: Optional = None
) -> Table:
    """Combine and filter transient candidates using KDTree + union-find clustering.
    
    Args:
        transients: Dictionary of transient tables from different catalogs
        min_catalogs: Minimum number of catalogs where source should be missing
        min_quality: Minimum quality score to include
        use_sky_coords: If True, use sky coordinates; if False, use pixel coordinates
        position_match_radius_arcsec: Matching radius in arcseconds (for sky coords)
        config: Optional configuration object
        
    Returns:
        Combined table of reliable transient candidates
    """
    if not transients:
        return Table()

    # Stack all candidates
    all_candidates = vstack(list(transients.values()))

    # Filter by quality
    quality_mask = all_candidates["quality_score"] >= min_quality
    all_candidates = all_candidates[quality_mask]

    if len(all_candidates) == 0:
        return Table()

    n_candidates = len(all_candidates)
    
    if use_sky_coords and 'ALPHA_J2000' in all_candidates.colnames and 'DELTA_J2000' in all_candidates.colnames:
        # Use sky coordinates with per-detection radii
        try:
            # Compute per-detection radii with configurable parameters
            nsigma = 3.0
            idlimit_min_px = 1.0 
            idlimit_max_px = 8.0
            
            if config:
                nsigma = getattr(config.detection, 'adaptive_nsigma', 3.0)
                idlimit_min_px = getattr(config.detection, 'idlimit_min_px', 1.0)
                idlimit_max_px = getattr(config.detection, 'idlimit_max_px', 8.0)
            
            per_det_radii = compute_per_detection_radius(
                all_candidates,
                nsigma=nsigma,
                idlimit_min_px=idlimit_min_px,
                idlimit_max_px=idlimit_max_px,
                config=config
            )
            
            # Convert to Cartesian coordinates for KDTree
            ra_rad = np.radians(all_candidates["ALPHA_J2000"])
            dec_rad = np.radians(all_candidates["DELTA_J2000"])
            
            x = np.cos(dec_rad) * np.cos(ra_rad)
            y = np.cos(dec_rad) * np.sin(ra_rad)
            z = np.sin(dec_rad)
            coords = np.column_stack((x, y, z))
            
            # Build KDTree
            tree = KDTree(coords)
            
            # Build adjacency graph using per-detection radii
            uf = UnionFind(n_candidates)
            
            # Use maximum possible radius for initial query, then filter
            max_radius_arcsec = max(position_match_radius_arcsec, np.max(per_det_radii))
            chord_length = 2 * np.sin(np.radians(max_radius_arcsec / 3600) / 2)
            
            neighbors_list = tree.query_radius(coords, r=chord_length)
            
            for i, neighbors in enumerate(neighbors_list):
                for j in neighbors:
                    if i < j:  # Avoid duplicate edges
                        # Calculate actual angular separation
                        coord_i = coords[i]
                        coord_j = coords[j]
                        
                        # Dot product for great circle distance
                        dot_product = np.clip(np.dot(coord_i, coord_j), -1.0, 1.0)
                        angular_sep_rad = np.arccos(dot_product)
                        angular_sep_arcsec = np.degrees(angular_sep_rad) * 3600
                        
                        # Check if within min of both detection radii
                        max_allowed_radius = min(per_det_radii[i], per_det_radii[j])
                        
                        if angular_sep_arcsec <= max_allowed_radius:
                            uf.union(i, j)
            
        except Exception as e:
            logging.warning(f"Sky coordinate clustering failed: {e}, falling back to pixel coordinates")
            use_sky_coords = False
    
    if not use_sky_coords:
        # Fallback to pixel coordinates
        coords = np.column_stack((all_candidates["X_IMAGE"], all_candidates["Y_IMAGE"]))
        tree = KDTree(coords)
        
        # Build adjacency graph with fixed pixel radius
        uf = UnionFind(n_candidates)
        pixel_radius = 2.0  # pixels
        
        neighbors_list = tree.query_radius(coords, r=pixel_radius)
        
        for i, neighbors in enumerate(neighbors_list):
            for j in neighbors:
                if i < j:  # Avoid duplicate edges
                    uf.union(i, j)
    
    # Get connected components
    components = uf.get_components()
    
    # Filter candidates appearing in enough catalogs and process components
    reliable = []
    
    for root, component_indices in components.items():
        # Count unique catalogs in this component
        component_data = all_candidates[component_indices]
        cat_count = len(set(component_data["reference_catalog"]))
        
        if cat_count >= min_catalogs:
            # Split component by epoch constraint if epoch information available
            if 'epoch_id' in all_candidates.colnames:
                subclusters = split_component_by_epoch(
                    component_indices, 
                    all_candidates, 
                    position_match_radius_arcsec
                )
            else:
                # No epoch info, treat as single cluster
                subclusters = [component_indices]
            
            # Process each subcluster
            for subcluster_indices in subclusters:
                subcluster_data = all_candidates[subcluster_indices]
                subcluster_cat_count = len(set(subcluster_data["reference_catalog"]))
                
                if subcluster_cat_count >= min_catalogs:
                    # Take the one with highest quality score from the subcluster
                    subcluster_qualities = subcluster_data["quality_score"]
                    best_local_idx = np.argmax(subcluster_qualities)
                    best_global_idx = subcluster_indices[best_local_idx]
                    
                    # Create single-row table instead of appending Row object
                    best_row = all_candidates[best_global_idx]
                    single_row_table = Table()
                    for col_name in best_row.colnames:
                        single_row_table[col_name] = [best_row[col_name]]
                    reliable.append(single_row_table)

    return vstack(reliable) if reliable else Table()

def compute_per_detection_radius(
    detections: Table,
    nsigma: float = 3.0,
    idlimit_min_px: float = 1.0,
    idlimit_max_px: float = 8.0,
    default_plate_scale_arcsec_per_px: float = 0.33,
    config: Optional = None
) -> np.ndarray:
    """
    Compute per-detection radius in arcseconds using PSF and SNR scaling.
    
    Args:
        detections: Table with detection data
        nsigma: Multiplier for PSF sigma (overridable by config.detection.adaptive_nsigma)
        idlimit_min_px: Minimum radius in pixels (overridable by config.detection.idlimit_min_px)
        idlimit_max_px: Maximum radius in pixels (overridable by config.detection.idlimit_max_px)
        default_plate_scale_arcsec_per_px: Default plate scale if WCS unavailable
        config: Optional config for parameter overrides
        
    Returns:
        Array of radii in arcseconds for each detection
    """
    # Override parameters from config if available
    if config:
        nsigma = getattr(config.detection, 'adaptive_nsigma', nsigma)
        idlimit_min_px = getattr(config.detection, 'idlimit_min_px', idlimit_min_px) 
        idlimit_max_px = getattr(config.detection, 'idlimit_max_px', idlimit_max_px)
        default_plate_scale_arcsec_per_px = getattr(config.detection, 'default_plate_scale_arcsec_per_px', default_plate_scale_arcsec_per_px)
    
    n_det = len(detections)
    radii_arcsec = np.full(n_det, 2.0)  # Default 2 arcsec
    
    try:
        # Get PSF proxy from FWHM_IMAGE
        if 'FWHM_IMAGE' in detections.colnames:
            fwhm_px = detections['FWHM_IMAGE']
            psf_sigma_px = fwhm_px / 2.35
        else:
            # Fallback: estimate from source size
            if 'A_IMAGE' in detections.colnames:
                psf_sigma_px = detections['A_IMAGE'] / 2.0
            else:
                psf_sigma_px = np.full(n_det, 2.0)  # Default 2 pixels
        
        # Get SNR proxy
        snr = np.full(n_det, 5.0)  # Default SNR
        if 'SNR' in detections.colnames:
            snr = np.maximum(detections['SNR'], 3.0)
        elif 'FLUX_AUTO' in detections.colnames and 'FLUXERR_AUTO' in detections.colnames:
            flux = detections['FLUX_AUTO']
            flux_err = detections['FLUXERR_AUTO']
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                snr = np.maximum(flux / np.maximum(flux_err, 1e-10), 3.0)
        
        # Scale radius by SNR (lower SNR = larger radius)
        snr_scale = np.sqrt(10.0 / snr)  # Scale factor
        
        # Calculate radius in pixels
        radii_px = nsigma * psf_sigma_px * snr_scale
        radii_px = np.clip(radii_px, idlimit_min_px, idlimit_max_px)
        
        # Convert to arcseconds
        plate_scale = default_plate_scale_arcsec_per_px
        
        # Try to get plate scale from WCS/header if available
        if hasattr(detections, 'meta') and detections.meta:
            # Check for CD matrix elements or CDELT
            cd11 = detections.meta.get('CD1_1', 0)
            cd22 = detections.meta.get('CD2_2', 0)
            if cd11 != 0 and cd22 != 0:
                plate_scale = 3600 * np.sqrt(abs(cd11 * cd22))
            else:
                cdelt1 = detections.meta.get('CDELT1', 0)
                if cdelt1 != 0:
                    plate_scale = 3600 * abs(cdelt1)
        
        radii_arcsec = radii_px * plate_scale
        
    except Exception as e:
        logging.warning(f"Error computing per-detection radii: {e}, using defaults")
        radii_arcsec = np.full(n_det, 2.0)
    
    return radii_arcsec


def split_component_by_epoch(
    component_indices: List[int],
    detections: Table,
    base_radius_arcsec: float = 2.0
) -> List[List[int]]:
    """
    Split a connected component to enforce one-per-epoch constraint.
    Uses greedy splitting with quality-based ordering.
    
    Args:
        component_indices: Indices of detections in this component
        detections: Full detection table
        base_radius_arcsec: Base radius for subcluster validation
        
    Returns:
        List of subclusters (each is a list of indices)
    """
    if len(component_indices) <= 1:
        return [component_indices]
    
    # Get component data
    component_data = detections[component_indices]
    
    # Sort by quality score descending (best first)
    if 'quality_score' in component_data.colnames:
        sorted_order = np.argsort(component_data['quality_score'])[::-1]
    else:
        # Fallback to magnitude (brighter = better)
        if 'MAG_AUTO' in component_data.colnames:
            sorted_order = np.argsort(component_data['MAG_AUTO'])
        else:
            sorted_order = np.arange(len(component_indices))
    
    sorted_indices = [component_indices[i] for i in sorted_order]
    sorted_data = component_data[sorted_order]
    
    # Initialize subclusters
    subclusters = []
    
    for i, idx in enumerate(sorted_indices):
        # Get epoch_id safely from table
        epoch_id = detections['epoch_id'][idx] if 'epoch_id' in detections.colnames else 0
        
        # Find first compatible subcluster
        placed = False
        for subcluster in subclusters:
            # Check epoch constraint
            subcluster_data = detections[subcluster]
            if 'epoch_id' in subcluster_data.colnames:
                subcluster_epochs = set(subcluster_data['epoch_id'])
                if epoch_id in subcluster_epochs:
                    continue  # Epoch conflict
            
            # Check distance constraint to subcluster centroid
            if len(subcluster) > 0:
                subcluster_positions = detections[subcluster]
                centroid_ra = np.mean(subcluster_positions['ALPHA_J2000'])
                centroid_dec = np.mean(subcluster_positions['DELTA_J2000'])
                
                # Calculate angular separation  
                detection_coord = SkyCoord(
                    ra=detections['ALPHA_J2000'][idx]*u.deg,
                    dec=detections['DELTA_J2000'][idx]*u.deg
                )
                centroid_coord = SkyCoord(
                    ra=centroid_ra*u.deg,
                    dec=centroid_dec*u.deg
                )
                
                separation = detection_coord.separation(centroid_coord).arcsec
                
                if separation <= base_radius_arcsec:
                    subcluster.append(idx)
                    placed = True
                    break
        
        # If not placed, start new subcluster
        if not placed:
            subclusters.append([idx])
    
    return subclusters


@dataclass
class HotPixelParams:
    """Parameters for hot pixel detection."""

    max_position_shift: float = 2.0  # Maximum allowed position shift in pixels
    min_detections: int = 3  # Minimum number of detections to consider
    max_flux_std: float = 0.2  # Maximum allowed standard deviation in normalized flux

class OptimizedTransientAnalyzer:
    """
    Enhanced TransientAnalyzer that uses optimized catalog functions.
    Drop-in replacement for your existing TransientAnalyzer.
    """

    def __init__(self, data_dir="/home/fnovotny/transient_work/", config=None) -> None:
        self.data_dir = Path(data_dir)
        self.config = config
        self.logger = logging.getLogger('transient_analyser.optimized')
        
        # Keep existing detection features
        self.det_features = {
            "basic": [
                "X_IMAGE", "Y_IMAGE", "FLUX_ISO", "FLUXERR_ISO",
                "MAG_ISO", "MAGERR_ISO",
            ],
            "shape": [
                "A_IMAGE", "B_IMAGE", "THETA_IMAGE", "ELONGATION",
                "ELLIPTICITY", "FWHM_IMAGE",
            ],
            "photometry": [
                "FLUX_MAX", "FLUX_AUTO", "MAG_AUTO", "KRON_RADIUS", "BACKGROUND",
            ],
            "quality": ["FLAGS", "SNR", "THRESHOLD", "ISOAREA_IMAGE"],
        }
        
        # Track loaded catalogs to avoid reloading
        self._loaded_catalogs = {}


    def find_transients_multicatalog(
        self,
        detections,
        catalogs,
        params=None,
        idlimit=5.0,
        radius_check=30.0,
        filter_pattern=None,
        gen_images=False,
        mag_change_threshold=1.0,
    ):
        """Enhanced version with better error handling."""
        results = {}
        vsx_early_applied = False

        for cat_name in catalogs:
            try:
                self.logger.info(f"Processing catalog: {cat_name}")
                
                # Try optimized path first with MAG_CALIB fallback
                try:
                    
                    # Prepare detections with magnitude fallback if needed
                    det_for_analysis = detections.copy()
                    if 'MAG_CALIB' not in det_for_analysis.colnames:
                        raise ValueError("No suitable magnitude/error columns available")
                    
                    catalog = self._get_optimized_catalog(cat_name, params)
                    
                    # Pass adaptive configuration through detections.meta if enabled
                    if self.config and self.config.detection.enable_adaptive_idlimit:
                        det_for_analysis.meta['adaptive_idlimit_enabled'] = True
                        det_for_analysis.meta['adaptive_nsigma'] = self.config.detection.adaptive_nsigma
                        det_for_analysis.meta['adaptive_percentile'] = self.config.detection.adaptive_percentile
                        det_for_analysis.meta['idlimit_min_px'] = self.config.detection.idlimit_min_px
                        det_for_analysis.meta['idlimit_max_px'] = self.config.detection.idlimit_max_px
                        det_for_analysis.meta['use_astvar'] = self.config.detection.use_astvar
                        
                        self.logger.debug(f"Enabled adaptive identification for {cat_name}: "
                                        f"nsigma={self.config.detection.adaptive_nsigma}, "
                                        f"percentile={self.config.detection.adaptive_percentile}%")
                    
                    candidates = catalog.get_transient_candidates_optimized(
                        detections=det_for_analysis,
                        idlimit=idlimit,
                        mag_change_threshold=mag_change_threshold,
                        siglim=5.0,
                        frame=10.0
                    )
                    self.logger.info(f"✅ Used optimized detection for {cat_name}")
                    
                except Exception as opt_error:
                    self.logger.warning(f"Optimized detection failed for {cat_name}: {opt_error}")
                    self.logger.info(f"Falling back to standard detection...")
                    
                    # Fallback to standard detection
                    if params is None:
                        params = QueryParams()
                    catalog = Catalog(catalog=cat_name, **params.__dict__)
                    
                    # Standard detection can handle various magnitude columns more flexibly
                    candidates = catalog.get_transient_candidates(detections, idlimit)
                    self.logger.info(f"✅ Used standard detection for {cat_name}")

                if len(candidates) > 0:
                    self.logger.info(f"Found {len(candidates)} candidates from {cat_name}")
                    
                    # Apply early VSX filter if enabled
                    if self.config and self.config.detection.vsx_filter_enabled:
                        try:
                            from catalog import filter_vsx_variables, CatalogCache
                            
                            # Initialize cache if not already done
                            if not hasattr(catalog, '_cache') or catalog._cache is None:
                                cache_dir = self.config.caching.cache_dir if self.config else "./catalog_cache"
                                catalog._cache = CatalogCache(cache_dir)
                            
                            # Apply VSX filter
                            self.logger.info(f"Applying VSX filter to {len(candidates)} candidates")
                            filtered_candidates, vsx_matches = filter_vsx_variables(
                                candidates,
                                catalog._cache,
                                match_radius_arcsec=self.config.detection.vsx_match_radius_arcsec,
                                catalog_id=self.config.detection.vsx_catalog_id
                            )
                            
                            # Update candidates with filtered results
                            candidates = filtered_candidates
                            self.logger.info(f"VSX filtering removed {len(vsx_matches)} known variables, {len(candidates)} candidates remain")
                            vsx_early_applied = True
                            
                        except Exception as vsx_error:
                            self.logger.warning(f"VSX filtering failed for {cat_name}: {vsx_error}")
                            self.logger.info("Continuing without VSX filtering...")

                    # Apply MAGLIM-based filtering: drop rows where MAG_CALIB > 1.1 * MAGLIM
                    try:
                        # Respect user's request: do not use MAG_ISO-substituted MAG_CALIB for this rule
                        if candidates.meta.get('mag_calib_is_fallback', False):
                            self.logger.debug("Skipping MAGLIM-based filtering (MAG_CALIB fallback was used)")
                        else:
                            maglim = None
                            for key in ('MAGLIM', 'MAGLIMIT', 'maglim', 'maglimit'):
                                if key in candidates.meta:
                                    maglim = float(candidates.meta[key])
                                    break
                            if maglim is not None and 'MAG_CALIB' in candidates.colnames:
                                keep_mask = np.array(candidates['MAG_CALIB'], dtype=float) <= (1.1 * maglim)
                                removed = int(np.sum(~keep_mask))
                                if removed > 0:
                                    candidates = candidates[keep_mask]
                                    self.logger.info(f"MAGLIM filter removed {removed} candidates (>1.5x MAGLIM), {len(candidates)} remain")
                    except Exception as e:
                        self.logger.debug(f"MAGLIM-based filtering skipped due to error: {e}")
                    
                    # Add features with error handling
                    try:
                        # Compute image_id for proper catalog context caching
                        image_id = catalog._generate_image_id(detections)
                        
                        self._add_detection_features(candidates)
                        self._add_catalog_context_safe(candidates, catalog, radius_check, filter_pattern, image_id=image_id)
                        self._add_quality_metrics(candidates)
                    except Exception as feature_error:
                        self.logger.warning(f"Feature addition failed for {cat_name}: {feature_error}")
                        # Ensure we have minimum required columns
                        if 'quality_score' not in candidates.columns:
                            candidates['quality_score'] = [0.5] * len(candidates)
                        if 'candidate_type' not in candidates.columns:
                            candidates['candidate_type'] = ['new'] * len(candidates)

                    candidates["reference_catalog"] = cat_name
                else:
                    self.logger.info(f"No candidates found from {cat_name}")
                    # Create empty table with required columns
                    candidates = Table()
                    candidates['quality_score'] = []
                    candidates['candidate_type'] = []
                    candidates["reference_catalog"] = []

                results[cat_name] = candidates

            except Exception as e:
                self.logger.error(f"Failed to process catalog {cat_name}: {str(e)}")
                # Create empty table with required columns for failed catalog
                empty_table = Table()
                empty_table['quality_score'] = []
                empty_table['candidate_type'] = []
                empty_table["reference_catalog"] = []
                results[cat_name] = empty_table
                continue

        # Apply late VSX filter fallback only if enabled and early filter did NOT run
        if (self.config and self.config.detection.vsx_filter_enabled and
            not vsx_early_applied and
            any(len(candidates) > 0 for candidates in results.values())):
            
            try:
                from catalog import filter_vsx_variables, CatalogCache
                
                self.logger.info("Applying late VSX filter fallback to final candidates")
                
                # Initialize cache
                cache_dir = self.config.caching.cache_dir if self.config else "./catalog_cache"
                cache = CatalogCache(cache_dir)
                
                # Apply VSX filter to each catalog's candidates
                total_filtered = 0
                for cat_name, candidates in results.items():
                    if len(candidates) > 0:
                        self.logger.debug(f"Applying late VSX filter to {len(candidates)} candidates from {cat_name}")
                        
                        filtered_candidates, vsx_matches = filter_vsx_variables(
                            candidates,
                            cache,
                            match_radius_arcsec=self.config.detection.vsx_match_radius_arcsec,
                            catalog_id=self.config.detection.vsx_catalog_id
                        )
                        
                        results[cat_name] = filtered_candidates
                        total_filtered += len(vsx_matches)
                
                if total_filtered > 0:
                    self.logger.info(f"Late VSX filter removed {total_filtered} additional known variables")
                else:
                    self.logger.debug("Late VSX filter found no additional variables to remove")
                    
            except Exception as vsx_error:
                self.logger.warning(f"Late VSX filtering failed: {vsx_error}")
                self.logger.info("Continuing with unfiltered candidates...")

        return results

    def _get_optimized_catalog(self, cat_name, params):
        """Get catalog with optimization, including error handling."""
        cache_key = f"{cat_name}_{hash(str(params.__dict__) if params else 'default')}"
        
        if cache_key in self._loaded_catalogs:
            self.logger.debug(f"Reusing loaded catalog: {cat_name}")
            return self._loaded_catalogs[cache_key]
        
        if params is None:
            params = QueryParams()
        
        self.logger.info(f"Loading catalog: {cat_name}")
        catalog = Catalog(catalog=cat_name, **params.__dict__)
        
        # Try to enable optimizations
        try:
            self.logger.debug(f"Precomputing photometric data for {cat_name}...")
            catalog.precompute_photometric_data()
            self.logger.debug(f"✅ Optimization enabled for {cat_name}")
        except Exception as e:
            self.logger.warning(f"Optimization failed for {cat_name}: {e}")
            self.logger.debug(f"Will use standard methods")
        
        self._loaded_catalogs[cache_key] = catalog
        return catalog

    def _add_catalog_context_safe(self, candidates, catalog, radius, filter_pattern=None, image_id=None):
        """Safe version of catalog context addition with fallbacks and config support."""
        if len(candidates) == 0:
            return
        
        # Use config values if available
        if self.config and hasattr(self.config.detection, 'radius_check'):
            radius = self.config.detection.radius_check
        
        try:
            # Try optimized method first
            positions = np.column_stack((candidates["X_IMAGE"], candidates["Y_IMAGE"]))
            stats = catalog.compute_local_statistics(
                positions=positions,
                radius=radius,
                filter_pattern=filter_pattern,
                image_id=image_id
            )
            
            for stat_name, values in stats.items():
                candidates[stat_name] = values
            self.logger.debug(f"✅ Used optimized context statistics")
            
        except Exception as e:
            self.logger.warning(f"Optimized context failed: {e}")
            # Add default values
            candidates["nearby_sources"] = [0] * len(candidates)
            candidates["source_density"] = [0.0] * len(candidates)
            candidates["nearest_source_dist"] = [np.inf] * len(candidates)
            self.logger.debug(f"✅ Added default context values")

    def _add_detection_features(self, candidates):
        """Add detection features using self.config if available."""
        if "A_IMAGE" in candidates.columns and "B_IMAGE" in candidates.columns:
            candidates["axis_ratio"] = candidates["B_IMAGE"] / candidates["A_IMAGE"]

        if "FWHM_IMAGE" in candidates.columns:
            median_fwhm = np.median(candidates["FWHM_IMAGE"])
            if median_fwhm > 0:
                candidates["fwhm_ratio"] = candidates["FWHM_IMAGE"] / median_fwhm
            else:
                candidates["fwhm_ratio"] = [1.0] * len(candidates)

        if "FLUX_AUTO" in candidates.columns and "FLUXERR_AUTO" in candidates.columns:
            candidates["snr_auto"] = candidates["FLUX_AUTO"] / np.maximum(candidates["FLUXERR_AUTO"], 1e-10)

        if "FLAGS" in candidates.columns:
            candidates["saturated"] = (candidates["FLAGS"] & 4) > 0
            candidates["blended"] = (candidates["FLAGS"] & 2) > 0
            candidates["near_bright"] = (candidates["FLAGS"] & 8) > 0

    def _add_quality_metrics(self, candidates):
        """Add quality metrics with safe defaults using self.config."""
        quality_score = np.ones(len(candidates))

        # Get config weights if available
        magnitude_weight = 1.0
        significance_weight = 2.0
        consistency_weight = 1.5
        isolation_weight = 1.0
        
        if self.config and hasattr(self.config.detection, 'magnitude_weight'):
            magnitude_weight = self.config.detection.magnitude_weight
            significance_weight = self.config.detection.significance_weight
            consistency_weight = self.config.detection.consistency_weight
            isolation_weight = self.config.detection.isolation_weight

        # Shape-based scores
        if "fwhm_ratio" in candidates.columns:
            quality_score *= np.exp(-((candidates["fwhm_ratio"] - 1) ** 2) / 0.5) * consistency_weight

        if "axis_ratio" in candidates.columns:
            quality_score *= np.clip(candidates["axis_ratio"], 0, 1) * consistency_weight

        # SNR-based score
        if "snr_auto" in candidates.columns:
            quality_score *= np.clip(candidates["snr_auto"] / 20.0, 0, 1) * significance_weight

        # Flag penalties
        if "FLAGS" in candidates.columns:
            quality_score *= np.where(candidates["FLAGS"] > 0, 0.5, 1.0)

        # Distance penalty
        if "nearest_source_dist" in candidates.columns:
            quality_score *= np.clip(candidates["nearest_source_dist"] / 10.0, 0, 1) * isolation_weight

        try:
            maglim = None
            for key in ('MAGLIM', 'MAGLIMIT', 'maglim', 'maglimit'):
                if key in getattr(candidates, 'meta', {}):
                    maglim = float(candidates.meta[key])
                    break
        except Exception:
            maglim = None

        if "MAG_CALIB" in candidates.columns and not candidates.meta.get('mag_calib_is_fallback', False):
            if maglim is not None:
                # Reward being brighter than the limiting magnitude
                delta = maglim - np.array(candidates["MAG_CALIB"], dtype=float)
                # Map delta to a bounded multiplier
                mag_factor = 1.0 + 0.2 * delta  # each mag brighter than limit boosts by 0.2
                mag_factor = np.clip(mag_factor, 0.5, 2.0) * magnitude_weight
                quality_score *= mag_factor
            else:
                # If no MAGLIM, apply a gentle brightness prior around 15 mag using MAG_CALIB only
                mag_factor = np.exp(-(np.array(candidates["MAG_CALIB"], dtype=float) - 15.0) / 3.0) * magnitude_weight
                quality_score *= np.clip(mag_factor, 0.1, 3.0)

        # Trail downweighting
        if "candidate_type" in candidates.columns:
            trail_downweight_factor = 3.0  # Default value
            if self.config and hasattr(self.config.detection, 'trail_downweight_factor'):
                trail_downweight_factor = self.config.detection.trail_downweight_factor
            
            # Apply downweight to trail candidates
            trail_mask = candidates["candidate_type"] == 'trail'
            if np.any(trail_mask):
                quality_score[trail_mask] /= trail_downweight_factor
                self.logger.debug(f"Downweighted {np.sum(trail_mask)} trail candidates by factor {trail_downweight_factor}")

        candidates["quality_score"] = quality_score
        candidates["quality_flag"] = np.where(
            quality_score > 0.8, "HIGH", np.where(quality_score > 0.5, "MEDIUM", "LOW")
        )

    def clear_catalog_cache(self) -> None:
        """Clear loaded catalogs to free memory."""
        for catalog in self._loaded_catalogs.values():
            catalog.clear_cache()
        self._loaded_catalogs.clear()
        self.logger.info("Catalog cache cleared")


class OptimizedMultiDetectionAnalyzer:
    """
    Enhanced MultiDetectionAnalyzer using optimized catalog functions.
    Drop-in replacement for your existing MultiDetectionAnalyzer.
    """

    def __init__(self, transient_analyzer: OptimizedTransientAnalyzer, lightcurve_dir="lightcurves", config=None):
        """
        Initialize the enhanced analyzer.
        
        Args:
            transient_analyzer: Instance of OptimizedTransientAnalyzer
            lightcurve_dir: Directory to save lightcurve data and plots
            config: Optional configuration object
        """
        self.transient_analyzer = transient_analyzer
        self.lightcurve_dir = Path(lightcurve_dir)
        self.config = config
        self.logger = logging.getLogger('transient_analyser.multi_detection')
        if not self.lightcurve_dir.exists():
            self.lightcurve_dir.mkdir(parents=True, exist_ok=True)

    def process_detection_tables_with_lightcurves(
        self,
        detection_tables: List[Table],
        catalogs: List[str],
        params: Optional[QueryParams] = None,
        idlimit: float = 5.0,
        radius_check: float = 30.0,
        filter_pattern: Optional[str] = None,
        min_catalogs: int = 1,
        min_quality: float = 0.1,
        position_match_radius: float = 2.0,
        min_n_detections: int = 3,
        mag_change_threshold: float = 1.0,  # New parameter
    ) -> Tuple[Table, Dict]:
        """
        Enhanced processing with optimized catalog operations and config support.
        
        Args:
            detection_tables: List of detection tables to process
            catalogs: List of catalog names to use
            params: Query parameters for catalogs
            idlimit: Identification limit in pixels
            radius_check: Radius for context checking in pixels
            filter_pattern: Pattern to match filters
            min_catalogs: Minimum number of catalogs where source should be missing
            min_quality: Minimum quality score to include
            position_match_radius: Radius for position matching in arcsec (overridden by config)
            min_n_detections: Minimum number of detections for valid transient
            mag_change_threshold: Magnitude change threshold for variability
            
        Returns:
            Tuple of (final_candidates_table, lightcurves_dict)
        """
        
        # Use config values if available
        if self.config:
            position_match_radius = getattr(self.config.detection, 'position_match_radius_arcsec', position_match_radius)
            min_n_detections = getattr(self.config.detection, 'min_n_detections', min_n_detections)
            min_catalogs = getattr(self.config.detection, 'min_catalogs', min_catalogs)
            min_quality = getattr(self.config.detection, 'min_quality', min_quality)
            
        self.logger.info(f"Using position_match_radius: {position_match_radius} arcsec from config")
        self.logger.info(f"Starting processing of {len(detection_tables)} detection tables...")
        
        # Step 1: Process each detection table efficiently
        self.logger.info("Step 1: Processing individual detection tables...")
        
        for i, det_table in enumerate(detection_tables):
            self.logger.debug(f"Processing detection table {i+1}/{len(detection_tables)}")
            
            # Use optimized transient analyzer
            transients = self.transient_analyzer.find_transients_multicatalog(
                det_table, 
                catalogs, 
                params, 
                idlimit, 
                radius_check, 
                filter_pattern,
                mag_change_threshold=mag_change_threshold
            )
            
            # Save results for this epoch
            self._save_epoch_results(transients, det_table, i, min_catalogs, min_quality)
        
        # Step 2: Enhanced cross-matching with lightcurve data collection
        self.logger.info("Step 2: Building lightcurves...")
        
        all_epoch_detections = self._prepare_epoch_detections(detection_tables)
        
        final_candidates, lightcurves = self._combine_with_lightcurves(
            detection_tables=detection_tables,
            all_epoch_detections=all_epoch_detections,
            position_match_radius=position_match_radius,
            min_n_detections=min_n_detections
        )
        
        # Step 3: Generate lightcurve plots and analysis
        if lightcurves:
            self.logger.info("Step 3: Generating lightcurve analysis...")
            self._analyze_and_plot_lightcurves(lightcurves)
            self._create_lightcurve_summary(lightcurves, final_candidates)
        
        self.logger.info(f"Final candidates: {len(final_candidates)}")
        self.logger.info(f"Lightcurves: {len(lightcurves)}")
        
        return final_candidates, lightcurves

    def _save_epoch_results(
        self, 
        transients: Dict[str, Table], 
        det_table: Table, 
        epoch_index: int,
        min_catalogs: int,
        min_quality: float
    ) -> None:
        """Save results for this epoch using existing combine_results function."""
        
        base_filename = self._get_base_filename(det_table, epoch_index)
        ecsv_table = f"{base_filename}_transients.ecsv"
        ecsv_path = self.transient_analyzer.data_dir / ecsv_table
        
        if not ecsv_path.exists():
            # Use enhanced combine_results function with KDTree + union-find
            config = getattr(self.transient_analyzer, 'config', None)
            reliable = combine_results(
                transients, 
                min_catalogs=min_catalogs, 
                min_quality=min_quality,
                use_sky_coords=True,
                position_match_radius_arcsec=2.0,
                config=config
            )
            reliable.write(str(ecsv_path), overwrite=True)
            self.logger.debug(f"Saved {len(reliable)} candidates to {ecsv_table}")

    def _prepare_epoch_detections(self, detection_tables: List[Table]) -> List[Table]:
        """Prepare epoch detection data with timing information."""
        all_epoch_detections = []
        
        for i, det_table in enumerate(detection_tables):
            # Extract timing information
            ctime = det_table.meta.get('CTIME', 0)
            exptime = det_table.meta.get('EXPTIME', 0)
            mid_time = ctime + exptime / 2.0
            
            # Add epoch information to detection table
            det_table_copy = det_table.copy()
            det_table_copy['epoch_id'] = i
            det_table_copy['obs_time'] = mid_time
            det_table_copy['mjd'] = self._unix_to_mjd(mid_time)
            det_table_copy['source_file'] = det_table.meta.get('filename', f'epoch_{i}')
            
            all_epoch_detections.append(det_table_copy)
        
        return all_epoch_detections

    def _combine_with_lightcurves(
        self,
        detection_tables: List[Table],
        all_epoch_detections: List[Table],
        position_match_radius: float = 2.0,
        min_n_detections: int = 3,
    ) -> Tuple[Table, Dict]:
        """Combine candidates and build lightcurves (same logic as before)."""
        
        logging.info(f"Starting lightcurve combination with {len(detection_tables)} epochs...")
        
        # Load all transient candidates
        all_candidates = []
        candidate_sources = []
        
        for i, det_table in enumerate(detection_tables):
            base_filename = self._get_base_filename(det_table, i)
            file_path = self.transient_analyzer.data_dir / f"{base_filename}_transients.ecsv"
            
            if file_path.exists():
                try:
                    candidates = Table.read(str(file_path))
                    if len(candidates) > 0:
                        candidates['epoch_id'] = i
                        candidates['source_file'] = base_filename
                        all_candidates.append(candidates)
                        candidate_sources.extend([i] * len(candidates))
                except Exception as e:
                    logging.info(f"Error reading {file_path}: {e}")
        
        if not all_candidates:
            return Table(), {}
        
        # Stack all candidates
        stacked_candidates = vstack(all_candidates, metadata_conflicts="silent")
        
        # Remove problematic columns
        if "quality_flag" in stacked_candidates.colnames:
            stacked_candidates.remove_column("quality_flag")
        
        # Filter out invalid coordinates early
        ra_values = stacked_candidates["ALPHA_J2000"]
        dec_values = stacked_candidates["DELTA_J2000"]
        
        # Check for NaN, infinite, or unrealistic coordinate values
        valid_coords = (
            np.isfinite(ra_values) & 
            np.isfinite(dec_values) & 
            (ra_values >= 0) & (ra_values <= 360) &
            (dec_values >= -90) & (dec_values <= 90)
        )
        
        if not np.all(valid_coords):
            n_invalid = np.sum(~valid_coords)
            logging.info(f"Filtering out {n_invalid} candidates with invalid coordinates")
            stacked_candidates = stacked_candidates[valid_coords]
            candidate_sources = [candidate_sources[i] for i in range(len(candidate_sources)) if valid_coords[i]]
        
        if len(stacked_candidates) == 0:
            logging.info("No candidates with valid coordinates remaining")
            return Table(), {}
            
        # Fast KDTree + union-find clustering with per-detection radii
        n_candidates = len(stacked_candidates)
        
        # Compute per-detection radii with configurable parameters
        config = getattr(self.transient_analyzer, 'config', None)
        
        nsigma = 3.0
        idlimit_min_px = 1.0
        idlimit_max_px = 8.0
        
        if config:
            nsigma = getattr(config.detection, 'adaptive_nsigma', 3.0)
            idlimit_min_px = getattr(config.detection, 'idlimit_min_px', 1.0)
            idlimit_max_px = getattr(config.detection, 'idlimit_max_px', 8.0)
        
        per_det_radii = compute_per_detection_radius(
            stacked_candidates,
            nsigma=nsigma,
            idlimit_min_px=idlimit_min_px,
            idlimit_max_px=idlimit_max_px,
            config=config
        )
        
        # Position matching using celestial coordinates
        ra_rad = np.radians(stacked_candidates["ALPHA_J2000"])
        dec_rad = np.radians(stacked_candidates["DELTA_J2000"])
        
        # Cartesian coordinates for efficient matching
        x = np.cos(dec_rad) * np.cos(ra_rad)
        y = np.cos(dec_rad) * np.sin(ra_rad)
        z = np.sin(dec_rad)
        coords = np.column_stack((x, y, z))
        
        # Build KDTree
        tree = KDTree(coords)
        
        # Two-pass approach: tight radius first, then optional merging
        # Pass 1: Build components with tight radius (0.6x base radius)
        tight_radius_arcsec = 0.6 * position_match_radius
        tight_chord_length = 2 * np.sin(np.radians(tight_radius_arcsec / 3600) / 2)
        
        logging.debug(f"Pass 1: Tight clustering with radius {tight_radius_arcsec:.2f} arcsec")
        
        # Build initial adjacency graph using per-detection radii
        uf = UnionFind(n_candidates)
        
        # Use maximum possible radius for initial query, then filter
        max_radius_arcsec = max(position_match_radius, np.max(per_det_radii))
        max_chord_length = 2 * np.sin(np.radians(max_radius_arcsec / 3600) / 2)
        
        neighbors_list = tree.query_radius(coords, r=max_chord_length)
        
        for i, neighbors in enumerate(neighbors_list):
            for j in neighbors:
                if i < j:  # Avoid duplicate edges
                    # Calculate actual angular separation
                    coord_i = coords[i]
                    coord_j = coords[j]
                    
                    # Dot product for great circle distance
                    dot_product = np.clip(np.dot(coord_i, coord_j), -1.0, 1.0)
                    angular_sep_rad = np.arccos(dot_product)
                    angular_sep_arcsec = np.degrees(angular_sep_rad) * 3600
                    
                    # Check if within min of both detection radii AND tight radius
                    max_allowed_radius = min(
                        per_det_radii[i], 
                        per_det_radii[j], 
                        tight_radius_arcsec
                    )
                    
                    if angular_sep_arcsec <= max_allowed_radius:
                        uf.union(i, j)
        
        # Get initial connected components
        components = uf.get_components()
        logging.debug(f"Pass 1: Found {len(components)} initial components")
        
        # Pass 2: Optional merging of components within base radius (with epoch check)
        component_centroids = {}
        component_epochs = {}
        
        for root, indices in components.items():
            component_data = stacked_candidates[indices]
            centroid_ra = np.mean(component_data['ALPHA_J2000'])
            centroid_dec = np.mean(component_data['DELTA_J2000'])
            component_centroids[root] = (centroid_ra, centroid_dec)
            component_epochs[root] = set(candidate_sources[i] for i in indices)
        
        # Check for mergeable components
        component_roots = list(components.keys())
        for i, root_i in enumerate(component_roots):
            for j in range(i+1, len(component_roots)):
                root_j = component_roots[j]
                
                # Skip if already merged
                if uf.find(root_i) == uf.find(root_j):
                    continue
                
                # Check centroid distance
                ra1, dec1 = component_centroids[root_i]
                ra2, dec2 = component_centroids[root_j]
                
                coord1 = SkyCoord(ra=ra1*u.deg, dec=dec1*u.deg)
                coord2 = SkyCoord(ra=ra2*u.deg, dec=dec2*u.deg)
                separation = coord1.separation(coord2).arcsec
                
                # Check if within base radius and no epoch conflicts
                if separation <= position_match_radius:
                    epochs_i = component_epochs[root_i]
                    epochs_j = component_epochs[root_j]
                    
                    if not (epochs_i & epochs_j):  # No epoch overlap
                        uf.union(root_i, root_j)
        
        # Get final components after merging
        final_components = uf.get_components()
        logging.info(f"Fast clustering: {len(final_components)} components from {n_candidates} candidates")
        
        # Process each component with epoch constraint enforcement
        final_candidates = []
        lightcurves = {}
        
        for root, component_indices in final_components.items():
            if len(component_indices) < min_n_detections:
                continue
            
            # Split component by epoch constraint
            subclusters = split_component_by_epoch(
                component_indices, 
                stacked_candidates, 
                position_match_radius
            )
            
            # Process each subcluster
            for subcluster_indices in subclusters:
                if len(subcluster_indices) < min_n_detections:
                    continue
                
                # Build lightcurve for this subcluster
                group_candidates = stacked_candidates[subcluster_indices]
                group_epochs = [candidate_sources[idx] for idx in subcluster_indices]
                
                try:
                    lightcurve_data = self._build_lightcurve_for_group(
                        group_candidates, group_epochs, all_epoch_detections, position_match_radius
                    )
                    
                    if len(lightcurve_data) >= min_n_detections:
                        # Create final candidate entry
                        best_local_idx = np.argmax(group_candidates["quality_score"])
                        best_global_idx = subcluster_indices[best_local_idx]
                        best_row = stacked_candidates[best_global_idx]
                        
                        # Convert to single-row table
                        best_candidate = Table()
                        for col_name in best_row.colnames:
                            best_candidate[col_name] = [best_row[col_name]]
                        
                        # Update with lightcurve statistics
                        self._update_candidate_with_lightcurve_stats(best_candidate, lightcurve_data)
                        
                        # Generate unique ID
                        transient_id = f"transient_{best_candidate['ALPHA_J2000'][0]:.3f}_{best_candidate['DELTA_J2000'][0]:.3f}"
                        best_candidate['transient_id'] = transient_id
                        
                        final_candidates.append(best_candidate)
                        lightcurves[transient_id] = lightcurve_data
                        
                except Exception as e:
                    logging.warning(f"Error processing subcluster: {e}")
                    continue
        
        # Convert to table
        if final_candidates:
            result_table = vstack(final_candidates)
            if "mag_range" in result_table.columns:
                try:
                    result_table["quality_score"] *= result_table["mag_range"]
                except Exception as e:
                    logging.info(f"Warning: Could not update quality_score: {e}")
                    result_table["quality_score"] = result_table["quality_score"] * result_table["mag_range"]
            result_table.sort("quality_score", reverse=True)
            
            # Trail detection summary logging
            if 'candidate_type' in result_table.colnames:
                trail_mask = result_table['candidate_type'] == 'trail'
                n_trails = np.sum(trail_mask)
                n_total = len(result_table)
                
                if n_trails > 0:
                    logging.info(f"Trail detection summary: {n_trails} trails out of {n_total} candidates ({100*n_trails/n_total:.1f}%)")
                    
                    # Log trail motion statistics
                    if 'motion_rate_as_per_hr' in result_table.colnames and 'motion_significance' in result_table.colnames:
                        trail_rates = result_table['motion_rate_as_per_hr'][trail_mask]
                        trail_sigs = result_table['motion_significance'][trail_mask]
                        
                        logging.info(f"Trail motion rates: {np.min(trail_rates):.2f}-{np.max(trail_rates):.2f}\"/hr "
                                   f"(median: {np.median(trail_rates):.2f}\"/hr)")
                        logging.info(f"Trail motion significance: {np.min(trail_sigs):.1f}-{np.max(trail_sigs):.1f} "
                                   f"(median: {np.median(trail_sigs):.1f})")
                else:
                    logging.info(f"Trail detection summary: no trails detected among {n_total} candidates")
        else:
            result_table = Table()
        
        return result_table, lightcurves

    # Keep all the existing methods for lightcurve building and analysis
    def _build_lightcurve_for_group(
        self, 
        group_candidates: Table, 
        group_epochs: List[int],
        all_epoch_detections: List[Table],
        match_radius: float
    ) -> Table:
        """Build lightcurve for a group (same as before)."""
        
        # Calculate mean position
        mean_ra = np.mean(group_candidates['ALPHA_J2000'])
        mean_dec = np.mean(group_candidates['DELTA_J2000'])
        target_coord = SkyCoord(ra=mean_ra*u.deg, dec=mean_dec*u.deg)
        
        # Collect matching detections
        all_detections = []
        
        for epoch_id, epoch_detections in enumerate(all_epoch_detections):
            if len(epoch_detections) == 0:
                continue
                
            try:
                ra_values = np.array(epoch_detections['ALPHA_J2000'], dtype=float)
                dec_values = np.array(epoch_detections['DELTA_J2000'], dtype=float)
                
                valid_mask = np.isfinite(ra_values) & np.isfinite(dec_values)
                if not np.any(valid_mask):
                    continue
                    
                ra_values = ra_values[valid_mask]
                dec_values = dec_values[valid_mask]
                valid_detections = epoch_detections[valid_mask]
                
                epoch_coords = SkyCoord(
                    ra=ra_values*u.deg,
                    dec=dec_values*u.deg
                )
                
                separations = target_coord.separation(epoch_coords)
                matches = separations < match_radius*u.arcsec
                
                if np.any(matches):
                    closest_idx = np.argmin(separations[matches])
                    matched_detection = valid_detections[matches][closest_idx]
                    all_detections.append(matched_detection)
                    
            except Exception as e:
                continue
        
        if all_detections:
            lightcurve = vstack(all_detections)
            lightcurve.sort('obs_time')
            return lightcurve
        else:
            return Table()

    def _get_base_filename(self, det_table: Table, epoch_index: int) -> str:
        """Get base filename with improved fallback logic."""
        filename = None
        for key in ['filename', 'FITSFILE', 'source_file', 'FILENAME']:
            if key in det_table.meta and det_table.meta[key]:
                filename = det_table.meta[key]
                break
        
        if filename is None:
            filename = f'epoch_{epoch_index}'
        
        # Clean filename
        filename = os.path.basename(str(filename))
        filename = os.path.splitext(filename)[0]
        
        # Remove problematic characters
        import re
        filename = re.sub(r'[^a-zA-Z0-9_\-]', '_', filename)
        filename = re.sub(r'_+', '_', filename).strip('_')
        
        return filename if filename else f'epoch_{epoch_index}'

    def _update_candidate_with_lightcurve_stats(self, candidate: Table, lightcurve: Table):
        """Update candidate with lightcurve statistics including motion/trail analysis."""
        
        # Time statistics
        time_span = (np.max(lightcurve['obs_time']) - np.min(lightcurve['obs_time'])) / 3600.0
        candidate['time_span_hours'] = time_span
        candidate['n_detections'] = len(lightcurve)
        candidate['n_epochs'] = len(np.unique(lightcurve['epoch_id']))
        
        # Position statistics
        ra_std = np.std(lightcurve['ALPHA_J2000']) * 3600
        dec_std = np.std(lightcurve['DELTA_J2000']) * 3600
        candidate['position_scatter_arcsec'] = np.sqrt(ra_std**2 + dec_std**2)
        
        # Motion analysis - compute motion features
        motion_features = self._compute_motion_features(lightcurve)
        for key, value in motion_features.items():
            candidate[key] = value
        
        # Trail features - analyze shape across epochs  
        trail_features = self._compute_trail_features(lightcurve, motion_features)
        for key, value in trail_features.items():
            candidate[key] = value
        
        # Trail scoring and decision
        trail_score, is_trail = self._compute_trail_score(motion_features, trail_features)
        candidate['trail_score'] = trail_score
        
        # Set candidate type based on trail analysis (unless already set to strong photometric event)
        # Read candidate_type safely from single-row table
        if 'candidate_type' in candidate.colnames and len(candidate['candidate_type']) > 0:
            ct_val = candidate['candidate_type'][0]
            current_type = ct_val[0] if isinstance(ct_val, (list, np.ndarray)) else ct_val
        else:
            current_type = 'unknown'
        if is_trail and current_type not in ['brightening', 'fading']:
            candidate['candidate_type'] = ['trail']
        elif 'candidate_type' not in candidate.colnames:
            candidate['candidate_type'] = ['unknown']
        
        # Photometric statistics
        if 'MAG_CALIB' in lightcurve.colnames:
            mags = lightcurve['MAG_CALIB']
            mag_errors = lightcurve['MAGERR_CALIB']
            
            
            candidate['mag_schizo'] = np.sum(np.sqrt(np.power(np.diff(mags),2)+np.power(np.diff(lightcurve['obs_time']),2)))

            # Weighted mean magnitude
            weights = 1.0 / mag_errors**2
            weighted_mean_mag = np.sum(mags * weights) / np.sum(weights)
            candidate['mag_weighted_mean'] = weighted_mean_mag
            
            # Magnitude variability metrics
            candidate['mag_range'] = np.max(mags) - np.min(mags)
            candidate['mag_std'] = np.std(mags)
            
            # Chi-squared test for variability
            chi2 = np.sum(((mags - weighted_mean_mag) / mag_errors)**2)
            reduced_chi2 = chi2 / (len(mags) - 1) if len(mags) > 1 else 0
            candidate['mag_chi2_reduced'] = reduced_chi2
            candidate['is_variable'] = reduced_chi2 > 2.0
            
            # Boost quality score for bright, variable objects
            brightness_factor = np.exp(-(weighted_mean_mag - 15.0) / 3.0)  # Brighter = higher score
            variability_factor = min(candidate['mag_range'][0] / 0.5, 3.0)  # Cap at 3x boost
            candidate['quality_score'] *= brightness_factor * variability_factor * self.config.detection.lc_shape_weight

        # Add strategy calculation
        self._add_strategy_fields(candidate, lightcurve)

    def _compute_motion_features(self, lightcurve: Table) -> Dict[str, float]:
        """
        Compute motion features per group based on linear fit to positions.
        
        Args:
            lightcurve: Table with ALPHA_J2000, DELTA_J2000, obs_time, epoch_id, SNR columns
            
        Returns:
            Dictionary of motion features
        """
        if len(lightcurve) < 2:
            return {
                'motion_ra_as_per_hr': 0.0,
                'motion_dec_as_per_hr': 0.0, 
                'motion_rate_as_per_hr': 0.0,
                'motion_sigma_as': 0.0,
                'motion_significance': 0.0,
                'n_epochs_moving': 0
            }
        
        # Choose reference epoch (highest SNR or median MJD)
        if 'SNR' in lightcurve.colnames:
            ref_idx = np.argmax(lightcurve['SNR'])
        elif 'mjd' in lightcurve.colnames:
            median_mjd = np.median(lightcurve['mjd'])
            ref_idx = np.argmin(np.abs(lightcurve['mjd'] - median_mjd))
        else:
            ref_idx = len(lightcurve) // 2
        
        ref_ra = lightcurve['ALPHA_J2000'][ref_idx]
        ref_dec = lightcurve['DELTA_J2000'][ref_idx]
        ref_time = lightcurve['obs_time'][ref_idx]
        
        # Convert times to hours relative to reference
        t_hr = (lightcurve['obs_time'] - ref_time) / 3600.0
        
        # Compute residuals in arcsec
        delta_ra_as = (lightcurve['ALPHA_J2000'] - ref_ra) * 3600.0 * np.cos(np.radians(ref_dec))
        delta_dec_as = (lightcurve['DELTA_J2000'] - ref_dec) * 3600.0
        
        # Linear fits for motion
        try:
            # RA motion: ΔRA_as = a_ra * t_hr + b_ra
            ra_coeffs = np.polyfit(t_hr, delta_ra_as, 1)
            a_ra = ra_coeffs[0]  # motion_ra_as_per_hr
            
            # Dec motion: ΔDec_as = a_dec * t_hr + b_dec  
            dec_coeffs = np.polyfit(t_hr, delta_dec_as, 1)
            a_dec = dec_coeffs[0]  # motion_dec_as_per_hr
            
            # Compute residuals from linear fits
            ra_pred = np.polyval(ra_coeffs, t_hr)
            dec_pred = np.polyval(dec_coeffs, t_hr)
            res_ra = delta_ra_as - ra_pred
            res_dec = delta_dec_as - dec_pred
            
            # Robust scatter via MAD/0.67
            mad_ra = np.median(np.abs(res_ra - np.median(res_ra))) / 0.67
            mad_dec = np.median(np.abs(res_dec - np.median(res_dec))) / 0.67
            motion_sigma_as = np.sqrt(mad_ra**2 + mad_dec**2)
            
            # Total motion rate
            motion_rate_as_per_hr = np.sqrt(a_ra**2 + a_dec**2)
            
            # Motion significance
            eps = 1e-6
            motion_significance = motion_rate_as_per_hr / max(motion_sigma_as, eps)
            
            # Count epochs following motion (residual norm <= 2 * sigma)
            res_norm = np.sqrt(res_ra**2 + res_dec**2)
            n_epochs_moving = np.sum(res_norm <= 2.0 * motion_sigma_as)
            
        except (np.linalg.LinAlgError, ValueError):
            # Fallback for degenerate cases
            a_ra = 0.0
            a_dec = 0.0 
            motion_rate_as_per_hr = 0.0
            motion_sigma_as = 0.0
            motion_significance = 0.0
            n_epochs_moving = 0
            
        return {
            'motion_ra_as_per_hr': a_ra,
            'motion_dec_as_per_hr': a_dec,
            'motion_rate_as_per_hr': motion_rate_as_per_hr,
            'motion_sigma_as': motion_sigma_as,
            'motion_significance': motion_significance,
            'n_epochs_moving': int(n_epochs_moving)
        }
    
    def _compute_trail_features(self, lightcurve: Table, motion_features: Dict = None) -> Dict[str, float]:
        """
        Compute trail cues from shape parameters across epochs.
        
        Args:
            lightcurve: Table with shape columns (A_IMAGE, B_IMAGE, THETA_IMAGE, FWHM_IMAGE)
            motion_features: Dictionary from _compute_motion_features (for alignment calculation)
            
        Returns:
            Dictionary of trail shape features
        """
        if len(lightcurve) == 0:
            return {'elongation_mean': 0.0, 'fwhm_ratio_mean': 1.0, 'align_mean': 0.0}
        
        # Per-epoch shape features
        elongations = []
        fwhm_ratios = []
        
        # Axis ratio and elongation
        if 'A_IMAGE' in lightcurve.colnames and 'B_IMAGE' in lightcurve.colnames:
            a_img = lightcurve['A_IMAGE']
            b_img = lightcurve['B_IMAGE']
            
            # Filter out invalid values
            valid_ab = (a_img > 0) & (b_img > 0) & np.isfinite(a_img) & np.isfinite(b_img)
            if np.any(valid_ab):
                axis_ratios = b_img[valid_ab] / a_img[valid_ab]
                elongations = 1.0 - axis_ratios
                elongations = np.clip(elongations, 0.0, 1.0)  # Ensure valid range
        
        # FWHM ratio relative to median
        if 'FWHM_IMAGE' in lightcurve.colnames:
            fwhm_vals = lightcurve['FWHM_IMAGE']
            valid_fwhm = (fwhm_vals > 0) & np.isfinite(fwhm_vals)
            
            if np.any(valid_fwhm):
                median_fwhm = np.median(fwhm_vals[valid_fwhm])
                if median_fwhm > 0:
                    fwhm_ratios = fwhm_vals[valid_fwhm] / median_fwhm
        
        # Compute means
        elongation_mean = np.mean(elongations) if len(elongations) > 0 else 0.0
        fwhm_ratio_mean = np.mean(fwhm_ratios) if len(fwhm_ratios) > 0 else 1.0
        
        # Alignment with motion direction
        align_mean = 0.0
        if (motion_features and 'THETA_IMAGE' in lightcurve.colnames and 
            motion_features.get('motion_rate_as_per_hr', 0) > 0):
            
            # Calculate motion direction angle (East of North)
            motion_ra = motion_features.get('motion_ra_as_per_hr', 0)
            motion_dec = motion_features.get('motion_dec_as_per_hr', 0)
            
            if motion_ra != 0 or motion_dec != 0:
                motion_angle_deg = np.degrees(np.arctan2(motion_ra, motion_dec))
                # Normalize to [0, 180] range for comparison with THETA_IMAGE
                if motion_angle_deg < 0:
                    motion_angle_deg += 180
                elif motion_angle_deg >= 180:
                    motion_angle_deg -= 180
                
                # Calculate alignment scores for each epoch
                theta_vals = lightcurve['THETA_IMAGE']
                valid_theta = np.isfinite(theta_vals)
                
                if np.any(valid_theta):
                    alignment_scores = []
                    for theta_img in theta_vals[valid_theta]:
                        # Calculate angular difference between motion and shape orientation
                        angle_diff = abs(motion_angle_deg - theta_img)
                        
                        # Handle wraparound at 180 degrees (position angles are symmetric)
                        if angle_diff > 90:
                            angle_diff = 180 - angle_diff
                        
                        # Convert to alignment score: 1.0 for perfect alignment (0°), 0.0 for perpendicular (90°)
                        alignment_score = 1.0 - (angle_diff / 90.0)
                        alignment_scores.append(alignment_score)
                    
                    align_mean = np.mean(alignment_scores) if alignment_scores else 0.0
        
        return {
            'elongation_mean': elongation_mean,
            'fwhm_ratio_mean': fwhm_ratio_mean, 
            'align_mean': align_mean
        }
    
    def _compute_trail_score(self, motion_features: Dict, trail_features: Dict) -> Tuple[float, bool]:
        """
        Compute trail score and decision based on motion and shape features.
        
        Args:
            motion_features: Dictionary from _compute_motion_features
            trail_features: Dictionary from _compute_trail_features
            
        Returns:
            Tuple of (trail_score, is_trail_decision)
        """
        # Get config parameters (use defaults if no config)
        if self.config:
            min_epochs = self.config.detection.trail_min_epochs
            motion_sigma_min = self.config.detection.trail_motion_sigma_min
            motion_sig_tau = self.config.detection.trail_motion_sig_tau
            score_threshold = self.config.detection.trail_score_threshold
        else:
            min_epochs = 3
            motion_sigma_min = 0.5
            motion_sig_tau = 3.0
            score_threshold = 0.7
        
        # Normalize components to [0,1] for scoring
        elongation_norm = np.clip(trail_features['elongation_mean'], 0.0, 1.0)
        
        # Motion significance normalized 
        motion_sig_norm = np.clip(motion_features['motion_significance'] / 10.0, 0.0, 1.0)
        
        # Alignment component (placeholder)
        align_norm = np.clip(trail_features['align_mean'], 0.0, 1.0)
        
        # Trail score with configurable weights (w1=0.4, w2=0.4, w3=0.2)
        w1, w2, w3 = 0.4, 0.4, 0.2
        trail_score = w1 * elongation_norm + w2 * motion_sig_norm + w3 * align_norm
        
        # Decision criteria
        n_epochs = motion_features['n_epochs_moving']
        motion_sigma = motion_features['motion_sigma_as']
        motion_significance = motion_features['motion_significance']
        
        is_trail = (
            n_epochs >= min_epochs and
            motion_sigma >= motion_sigma_min and 
            motion_significance >= motion_sig_tau and
            trail_score >= score_threshold
        )
        
        # Debug logging for trail-tagged candidates
        if is_trail and hasattr(self, 'logger'):
            self.logger.debug(f"Trail candidate: motion_rate={motion_features['motion_rate_as_per_hr']:.2f}\"/hr, "
                            f"motion_sigma={motion_sigma:.2f}\", motion_sig={motion_significance:.1f}, "
                            f"n_epochs={n_epochs}, trail_score={trail_score:.3f}")
        
        return trail_score, is_trail

    def _add_strategy_fields(self, candidate: Table, lightcurve: Table):
        """
        Add strategy calculation fields to candidate based on previous observation.
        
        Args:
            candidate: Single-row candidate table to update
            lightcurve: Full lightcurve table for this candidate
        """
        # Initialize all strategy fields with null values
        strategy_fields = {
            'strategy_config': None,
            'strategy_exp_s': None,
            'strategy_snr': None,
            'strategy_filters': None,
            'strategy_emccd': None,
            'strategy_prev_frame': None,
            'strategy_ecsv': None,
            'strategy_time_since_trigger_s': None,
            'strategy_sky_1s': None,
            'strategy_fwhm_px': None,
            'strategy_magzero_1s': None
        }
        
        try:
            # Require at least 2 lightcurve points
            if len(lightcurve) < 2:
                for field, value in strategy_fields.items():
                    candidate[field] = [value]
                return
            
            # Sort lightcurve by observation time
            if 'obs_time' in lightcurve.colnames:
                sorted_lc = lightcurve[np.argsort(lightcurve['obs_time'])]
            elif 'mjd' in lightcurve.colnames:
                sorted_lc = lightcurve[np.argsort(lightcurve['mjd'])]
            else:
                # Use epoch_id as fallback
                sorted_lc = lightcurve[np.argsort(lightcurve['epoch_id'])]
            
            # Get indices for latest (L) and previous (P = L-1) observations
            latest_idx = len(sorted_lc) - 1
            prev_idx = latest_idx - 1
            
            prev_row = sorted_lc[prev_idx]
            
            # Get magnitude from previous observation
            if 'MAG_CALIB' not in prev_row.colnames:
                self.logger.debug("No MAG_CALIB in previous observation, skipping strategy calculation")
                for field, value in strategy_fields.items():
                    candidate[field] = [value]
                return
            
            magnitude = float(prev_row['MAG_CALIB'])
            
            # Calculate time since trigger
            time_since_trigger_s = None
            
            # Try to get GRB T0 from config or metadata
            grb_t0 = None
            if self.config and hasattr(self.config, 'grb_t0'):
                grb_t0 = self.config.grb_t0
            elif hasattr(sorted_lc, 'meta') and 'grb_t0' in sorted_lc.meta:
                grb_t0 = sorted_lc.meta['grb_t0']
            
            if grb_t0 is not None and 'obs_time' in prev_row.colnames:
                time_since_trigger_s = float(prev_row['obs_time']) - grb_t0
            else:
                # Fallback: relative time from first observation
                if 'obs_time' in sorted_lc.colnames:
                    first_time = float(sorted_lc['obs_time'][0])
                    prev_time = float(prev_row['obs_time'])
                    time_since_trigger_s = prev_time - first_time
                else:
                    time_since_trigger_s = 3600.0  # Default 1 hour
            
            # Get previous frame's source file and derive ECSV path
            prev_ecsv_path = None
            if 'source_file' in prev_row.colnames:
                source_file = str(prev_row['source_file'])
                strategy_fields['strategy_prev_frame'] = source_file
                
                # Try to find corresponding ECSV file
                data_dir = self.transient_analyzer.data_dir
                
                # Method 1: Direct match
                ecsv_candidate = data_dir / f"{source_file}.ecsv"
                if ecsv_candidate.exists():
                    prev_ecsv_path = str(ecsv_candidate)
                else:
                    # Method 2: Glob search
                    import glob
                    glob_pattern = str(data_dir / f"{source_file}*.ecsv")
                    matches = glob.glob(glob_pattern)
                    if matches:
                        prev_ecsv_path = matches[0]
                    else:
                        # Method 3: Fallback to image.ecsv
                        fallback_path = data_dir / "image.ecsv"
                        if fallback_path.exists():
                            prev_ecsv_path = str(fallback_path)
            
            if prev_ecsv_path is None:
                self.logger.debug("Could not find ECSV file for previous frame, skipping strategy calculation")
                for field, value in strategy_fields.items():
                    candidate[field] = [value]
                return
            
            strategy_fields['strategy_ecsv'] = prev_ecsv_path
            strategy_fields['strategy_time_since_trigger_s'] = time_since_trigger_s
            
            # Import and call strategy calculator
            try:
                from strategy_v2 import determine_grb_strategy
                
                strategy_result = determine_grb_strategy(
                    magnitude=magnitude,
                    time_since_trigger=time_since_trigger_s,
                    ecsv_file=prev_ecsv_path
                )
                
                # Extract strategy results
                strategy_fields['strategy_config'] = strategy_result.get('config_name')
                strategy_fields['strategy_exp_s'] = strategy_result.get('exp_time')
                strategy_fields['strategy_snr'] = strategy_result.get('snr')
                strategy_fields['strategy_filters'] = strategy_result.get('num_filters')
                strategy_fields['strategy_emccd'] = strategy_result.get('use_emccd')
                strategy_fields['strategy_magzero_1s'] = strategy_result.get('magzero_1s')
                
                # Extract background conditions if available
                if 'background_conditions' in strategy_result:
                    bg_conditions = strategy_result['background_conditions']
                    # Parse conditions string like 'sky_1s=11.5 ph/s/px, FWHM=2.1px'
                    try:
                        import re
                        sky_match = re.search(r'sky_1s=([0-9.]+)', bg_conditions)
                        fwhm_match = re.search(r'FWHM=([0-9.]+)', bg_conditions)
                        
                        if sky_match:
                            strategy_fields['strategy_sky_1s'] = float(sky_match.group(1))
                        if fwhm_match:
                            strategy_fields['strategy_fwhm_px'] = float(fwhm_match.group(1))
                    except Exception as parse_error:
                        self.logger.debug(f"Could not parse background conditions: {parse_error}")
                
                self.logger.debug(f"Strategy calculated: {strategy_result.get('config_name')} "
                                f"({strategy_result.get('exp_time'):.1f}s, SNR={strategy_result.get('snr'):.1f})")
                
            except ImportError as e:
                self.logger.warning(f"Could not import strategy_v2: {e}")
            except Exception as e:
                self.logger.warning(f"Strategy calculation failed: {e}")
        
        except Exception as e:
            self.logger.debug(f"Error in strategy field calculation: {e}")
        
        finally:
            # Always add all fields (with None values if calculation failed)
            for field, value in strategy_fields.items():
                candidate[field] = [value]

    def _unix_to_mjd(self, unix_time):
        """Convert Unix timestamp to Modified Julian Date."""
        try:
            t = Time(unix_time, format='unix')
            return t.mjd
        except:
            return unix_time

    # Keep all existing plotting methods unchanged
    def _analyze_and_plot_lightcurves(self, lightcurves: Dict):
        """Generate lightcurve plots and analysis."""
        for transient_id, lightcurve in lightcurves.items():
            self._plot_individual_lightcurve(transient_id, lightcurve)
            self._save_lightcurve_data(transient_id, lightcurve)

    def _plot_individual_lightcurve(self, transient_id: str, lightcurve: Table):
        """Create detailed lightcurve plot."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        times = lightcurve['obs_time']
        time_hours = (times - times[0]) / 3600.0
        
        if 'MAG_CALIB' in lightcurve.colnames:
            mags = lightcurve['MAG_CALIB']
            mag_errs = lightcurve['MAGERR_CALIB']
            ax.set_ylabel('Calibrated Magnitude')
        else:
            mags = lightcurve['MAG_ISO']
            mag_errs = lightcurve['MAGERR_ISO']
            ax.set_ylabel('Instrumental Magnitude')
        
        ax.errorbar(time_hours, mags, yerr=mag_errs, fmt='o-', capsize=3, markersize=6)
        ax.invert_yaxis()
        ax.set_xlabel('Time since first detection (hours)')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Lightcurve: {transient_id}')
        
        # Add statistics
        stats_text = self._generate_stats_text(lightcurve)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plot_filename = self.lightcurve_dir / f"{transient_id}_lightcurve.png"
        plt.savefig(str(plot_filename), dpi=150, bbox_inches='tight')
        plt.close()

    def _generate_stats_text(self, lightcurve: Table) -> str:
        """Generate statistics text for lightcurve plot."""
        stats = []
        stats.append(f"N detections: {len(lightcurve)}")
        stats.append(f"N epochs: {len(np.unique(lightcurve['epoch_id']))}")
        
        time_span = (np.max(lightcurve['obs_time']) - np.min(lightcurve['obs_time'])) / 3600.0
        stats.append(f"Time span: {time_span:.1f} hours")
        
        if 'MAG_CALIB' in lightcurve.colnames:
            mags = lightcurve['MAG_CALIB']
            stats.append(f"Mag range: {np.max(mags) - np.min(mags):.2f}")
            stats.append(f"Mag std: {np.std(mags):.3f}")
        
        ra_std = np.std(lightcurve['ALPHA_J2000']) * 3600
        dec_std = np.std(lightcurve['DELTA_J2000']) * 3600
        pos_scatter = np.sqrt(ra_std**2 + dec_std**2)
        stats.append(f"Pos scatter: {pos_scatter:.2f}\"")
        
        return '\n'.join(stats)

    def _save_lightcurve_data(self, transient_id: str, lightcurve: Table):
        """Save lightcurve data to file."""
        lightcurve.meta['transient_id'] = transient_id
        lightcurve.meta['n_detections'] = len(lightcurve)
        lightcurve.meta['n_epochs'] = len(np.unique(lightcurve['epoch_id']))
        
        filename = self.lightcurve_dir / f"{transient_id}_lightcurve.ecsv"
        lightcurve.write(str(filename), format='ascii.ecsv', overwrite=True)
    
    def _create_lightcurve_summary(self, lightcurves: Dict, final_candidates: Table):
        """Create summary plots and analysis."""
        
        # Summary grid plot
        self._create_summary_grid_plot(lightcurves, final_candidates)
        

    def _create_summary_grid_plot(self, lightcurves: Dict, final_candidates, max_plots=32):
        """
        Enhanced grid plot with color coding for different candidate types.
        """
        
        n_plots = min(len(lightcurves), max_plots)
        if n_plots == 0:
            return
            
        cols = int(np.ceil(np.sqrt(n_plots)))
        rows = int(np.ceil(n_plots / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
        if n_plots == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        axes_flat = axes.flatten() if n_plots > 1 else axes
        
        # Define colors for different candidate types
        type_colors = {
            'new': 'blue',
            'brightening': 'red', 
            'fading': 'orange',
            'trail': 'purple',
            'unknown': 'gray'
        }
        
        type_symbols = {
            'new': 'o',
            'brightening': '^',  # Triangle up
            'fading': 'v',       # Triangle down
            'trail': 'D',        # Diamond
            'unknown': 's'       # Square
        }
        
        # Sort lightcurves by quality score (highest first)
        lightcurve_items = []
        for transient_id, lc in lightcurves.items():
            candidate_mask = final_candidates['transient_id'] == transient_id
            if np.any(candidate_mask):
                quality_score = final_candidates[candidate_mask][0]['quality_score']
                lightcurve_items.append((quality_score, transient_id, lc))
            else:
                lightcurve_items.append((0.0, transient_id, lc))
        
        # Sort by quality score (descending)
        lightcurve_items.sort(key=lambda x: x[0], reverse=True)
        
        for i, (quality_score, transient_id, lc) in enumerate(lightcurve_items[:max_plots]):
            ax = axes_flat[i]
            
            times = lc['obs_time']
            time_hours = (times - times[0]) / 3600.0
            
            if 'MAG_CALIB' in lc.colnames:
                mags = lc['MAG_CALIB']
                mag_errs = lc['MAGERR_CALIB']
            else:
                mags = lc['MAG_ISO']
                mag_errs = lc['MAGERR_ISO']
            
            # Get candidate type for color coding
            candidate_mask = final_candidates['transient_id'] == transient_id
            if np.any(candidate_mask):
                candidate = final_candidates[candidate_mask][0]
                candidate_type = candidate['candidate_type'] if 'candidate_type' in candidate.colnames else 'unknown'
            else:
                candidate_type = 'unknown'
            
            # Plot with appropriate color and symbol
            color = type_colors.get(candidate_type, 'gray')
            symbol = type_symbols.get(candidate_type, 'o')
            
            ax.errorbar(time_hours, mags, yerr=mag_errs, 
                    fmt=f'{symbol}-', color=color, markersize=4, capsize=2,
                    label=candidate_type.capitalize())
            
            ax.set_title(f"{transient_id[:15]}...\nN={len(lc)} ({candidate_type})", 
                        fontsize=8)
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=6)
        
        # Hide unused subplots
        for i in range(n_plots, len(axes_flat)):
            axes_flat[i].set_visible(False)
        
        # Add legend
        if n_plots > 0:
            # Create custom legend
            from matplotlib.lines import Line2D
            legend_elements = []
            for ctype, color in type_colors.items():
                symbol = type_symbols[ctype]
                legend_elements.append(Line2D([0], [0], marker=symbol, color=color, 
                                            linestyle='-', markersize=6, label=ctype.capitalize()))
            
            fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
        
        plt.tight_layout()
        save_path = self.lightcurve_dir / 'lightcurves_summary.png'
        plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
        plt.close()        
