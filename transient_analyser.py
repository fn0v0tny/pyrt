import os
from pathlib import Path
import warnings
from typing import Dict, List, Optional, Set

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


@dataclass
class HotPixelParams:
    """Parameters for hot pixel detection."""

    max_position_shift: float = 2.0  # Maximum allowed position shift in pixels
    min_detections: int = 3  # Minimum number of detections to consider
    max_flux_std: float = 0.2  # Maximum allowed standard deviation in normalized flux


class TransientAnalyzer:
    """Combined system for transient detection and feature extraction."""

    def __init__(self, data_dir="/home/fnovotny/transient_work/") -> None:
        # Detection features we want to extract if available
        self.data_dir = Path(data_dir)
        self.det_features = {
            "basic": [
                "X_IMAGE",
                "Y_IMAGE",
                "FLUX_ISO",
                "FLUXERR_ISO",
                "MAG_ISO",
                "MAGERR_ISO",
            ],
            "shape": [
                "A_IMAGE",
                "B_IMAGE",
                "THETA_IMAGE",
                "ELONGATION",
                "ELLIPTICITY",
                "FWHM_IMAGE",
            ],
            "photometry": [
                "FLUX_MAX",
                "FLUX_AUTO",
                "MAG_AUTO",
                "KRON_RADIUS",
                "BACKGROUND",
            ],
            "quality": ["FLAGS", "SNR", "THRESHOLD", "ISOAREA_IMAGE"],
        }

    def find_transients_multicatalog(
        self,
        detections: Table,
        catalogs: List[str],
        params: Optional[QueryParams] = None,
        idlimit: float = 5.0,
        radius_check: float = 30.0,
        filter_pattern: Optional[str] = None,
        gen_images: bool = False,
    ) -> Dict[str, Table]:
        """Find and analyze transient candidates using multiple catalogs."""
        results = {}

        for cat_name in catalogs:
            try:
                # Create catalog instance
                if params is None:
                    params = QueryParams()
                catalog = Catalog(catalog=cat_name, **params.__dict__)

                # Get candidates and enrich with features
                candidates = catalog.get_transient_candidates(detections, idlimit)

                if len(candidates) > 0:
                    self._add_detection_features(candidates)
                    self._add_catalog_context(
                        candidates, catalog, radius_check, filter_pattern
                    )
                    self._add_quality_metrics(candidates)

                    # Add catalog identifier
                    candidates["reference_catalog"] = cat_name

                results[cat_name] = candidates
                # Generate images for each candidate
                del catalog
            except Exception as e:
                warnings.warn(f"Failed to process catalog {cat_name}: {str(e)}")
                continue
        return results
    
    def _add_detection_features(self, candidates: Table) -> None:
        """Add detection-based features to candidate table."""
        # Shape features
        if all(f in candidates.columns for f in ["A_IMAGE", "B_IMAGE"]):
            candidates["axis_ratio"] = candidates["B_IMAGE"] / candidates["A_IMAGE"]

        if "FWHM_IMAGE" in candidates.columns:
            median_fwhm = np.median(candidates["FWHM_IMAGE"])
            candidates["fwhm_ratio"] = candidates["FWHM_IMAGE"] / median_fwhm

        # Photometric featuresdata_dir
        if all(f in candidates.columns for f in ["FLUX_AUTO", "FLUXERR_AUTO"]):
            candidates["snr_auto"] = (
                candidates["FLUX_AUTO"] / candidates["FLUXERR_AUTO"]
            )

        # Magnitude differences
        mag_pairs = [
            ("MAG_AUTO", "MAG_ISO"),
            ("MAG_AUTO", "MAG_APER"),
            ("MAG_ISO", "MAG_APER"),
        ]

        for mag1, mag2 in mag_pairs:
            if mag1 in candidates.columns and mag2 in candidates.columns:
                candidates[f"{mag1}_{mag2}_diff"] = candidates[mag1] - candidates[mag2]

        # Flag interpretation
        if "FLAGS" in candidates.columns:
            candidates["saturated"] = (candidates["FLAGS"] & 4) > 0
            candidates["blended"] = (candidates["FLAGS"] & 2) > 0
            candidates["near_bright"] = (candidates["FLAGS"] & 8) > 0

    def _add_catalog_context(
        self,
        candidates: Table,
        catalog: Catalog,
        radius: float,
        filter_pattern: Optional[str] = None,
    ) -> None:
        """Add contextual information from catalog.
        
        Args:
            candidates: Candidate table
            catalog: Reference catalog
            radius: Search radius in pixels
            filter_pattern: Pattern to match filters (e.g., 'R' will match 'R1', 'R2', etc.)
        """
        cat_xy = catalog._transform_catalog_to_pixel(candidates)
        if len(cat_xy) == 0:
            return

        cand_xy = np.column_stack((candidates["X_IMAGE"], candidates["Y_IMAGE"]))

        # Find sources within radius
        tree = KDTree(cat_xy)
        neighbors = tree.query_radius(cand_xy, r=radius)
        distances, _ = tree.query(cand_xy, k=1)

        # Density features
        candidates["nearby_sources"] = [len(n) for n in neighbors]
        candidates["source_density"] = candidates["nearby_sources"] / (
            np.pi * radius ** 2
        )
        candidates["nearest_source_dist"] = distances.flatten()

        # Filter-specific features
        if hasattr(catalog, "filters") and filter_pattern is not None:
            # Get matching filters
            matching_filters = [
                filter_key
                for filter_key, filter_obj in catalog.filters.items()
                if filter_pattern.upper() in filter_key.upper()
            ]
            print(matching_filters)
            for filter_key in matching_filters[:1]:
                filter_obj = catalog.filters[filter_key]
                col_name = filter_obj.name  # Get actual column name
                print(col_name)
                if col_name in catalog.columns:
                    # Get magnitudes for all neighbors
                    mags = [catalog[col_name][n] for n in neighbors]

                    # Compute statistics using the filter key for naming
                    candidates[f"mean_mag_{filter_pattern}"] = [
                        np.mean(m) if len(m) > 0 else np.nan for m in mags
                    ]
                    candidates[f"std_mag_{filter_pattern}"] = [
                        np.std(m) if len(m) > 0 else np.nan for m in mags
                    ]

                    # Add error information if available
                    if (
                        filter_obj.error_name
                        and filter_obj.error_name in catalog.columns
                    ):
                        errs = [catalog[filter_obj.error_name][n] for n in neighbors]
                        candidates[f"mean_err_{filter_pattern}"] = [
                            np.mean(e) if len(e) > 0 else np.nan for e in errs
                        ]

    def _add_quality_metrics(self, candidates: Table) -> None:
        """Add computed quality metrics."""
        quality_score = np.ones(len(candidates))

        # Shape-based scores
        if "fwhm_ratio" in candidates.columns:
            quality_score *= np.exp(-((candidates["fwhm_ratio"] - 1) ** 2) / 0.5)

        if "axis_ratio" in candidates.columns:
            quality_score *= np.clip(candidates["axis_ratio"], 0, 1)

        # SNR-based score
        if "snr_auto" in candidates.columns:
            quality_score *= np.clip(candidates["snr_auto"] / 20.0, 0, 1)

        # Flag penalties
        if "FLAGS" in candidates.columns:
            quality_score *= np.where(candidates["FLAGS"] > 0, 0.5, 1.0)

        # Distance penalty
        if "nearest_source_dist" in candidates.columns:
            quality_score *= np.clip(candidates["nearest_source_dist"] / 10.0, 0, 1)

        candidates["quality_score"] = quality_score
        candidates["quality_flag"] = np.where(
            quality_score > 0.8, "HIGH", np.where(quality_score > 0.5, "MEDIUM", "LOW")
        )


def combine_results(
    transients: Dict[str, Table], min_catalogs: int = 1, min_quality: float = 0.5
) -> Table:
    """Combine and filter transient candidates from multiple catalogs.
    
    Args:
        transients: Dictionary of transient tables from different catalogs
        min_catalogs: Minimum number of catalogs where source should be missing
        min_quality: Minimum quality score to include
        
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

    # Group by position
    coords = np.column_stack((all_candidates["X_IMAGE"], all_candidates["Y_IMAGE"]))
    tree = KDTree(coords)

    # Find groups within small radius
    groups = tree.query_radius(coords, r=2.0)  # 2 pixel radius for grouping

    # Filter candidates appearing in enough catalogs
    reliable = []
    processed = set()

    for i, group in enumerate(groups):
        if i in processed:
            continue

        # Count unique catalogs
        cat_count = len({all_candidates[idx]["reference_catalog"] for idx in group})

        if cat_count >= min_catalogs:
            # Take the one with highest quality score from the group
            group_qualities = all_candidates["quality_score"][group]
            best_idx = group[np.argmax(group_qualities)]
            reliable.append(all_candidates[best_idx])
            processed.update(group)

    return vstack(reliable) if reliable else Table()


class MultiDetectionAnalyzer:
    # Metadata keys that we want to preserve as columns
    IMPORTANT_METADATA = {}
    """Analyzer for processing transients across multiple detection tables."""

    def __init__(self, transient_analyzer, lightcurve_dir="lightcurves"):
        """
        Initialize the enhanced analyzer.
        
        Args:
            transient_analyzer: Instance of TransientAnalyzer
            lightcurve_dir: Directory to save lightcurve data and plots
        """
        self.transient_analyzer = transient_analyzer
        self.lightcurve_dir = Path(lightcurve_dir)
        if not self.lightcurve_dir.exists():
            self.lightcurve_dir.mkdir(parents=True, exist_ok=True)

        self.all_detections = []
        self.detection_metadata = []


    def process_detection_tables_with_lightcurves(
    self,
    detection_tables: List[Table],
    catalogs: List[str],
    params: Optional['QueryParams'] = None,
    idlimit: float = 5.0,
    radius_check: float = 30.0,
    filter_pattern: Optional[str] = None,
    min_catalogs: int = 1,
    min_quality: float = 0.1,
    position_match_radius: float = 2.0,
    min_n_detections: int = 3
) -> tuple[Table, Dict]:
        """Enhanced version that generates lightcurves during the analysis process."""
        print("Processing detection tables with lightcurve generation...")
        
        # Step 1: Collect all detections with metadata
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
        
        # Step 2: Find transients in each epoch with FIXED filename logic
        for i, det_table in enumerate(detection_tables):
            # FIXED: Better filename handling using multiple metadata sources
            base_filename = self._get_base_filename(det_table, i)
            ecsv_table = f"{base_filename}_transients.ecsv"
            ecsv_path = self.transient_analyzer.data_dir / ecsv_table
            
            if not ecsv_path.exists():
                print(f"Processing epoch {i+1}/{len(detection_tables)}: {ecsv_table}")
                from transient_analyser import combine_results
                transients = self.transient_analyzer.find_transients_multicatalog(
                    det_table, catalogs, params, idlimit, radius_check, filter_pattern
                )
                reliable = combine_results(transients, min_catalogs=min_catalogs, min_quality=min_quality)
                reliable.write(str(ecsv_path), overwrite=True)
                print(f"Saved {len(reliable)} candidates to {ecsv_table}")
            else:
                print(f"Using existing transients file: {ecsv_table}")
        
        # Step 3: Enhanced cross-matching with lightcurve data collection
        final_candidates, lightcurves = self._combine_with_lightcurves(
            detection_tables=detection_tables,
            all_epoch_detections=all_epoch_detections,
            position_match_radius=position_match_radius,
            min_n_detections=min_n_detections
        )
        
        # Step 4: Generate lightcurve plots and analysis
        if lightcurves:
            self._analyze_and_plot_lightcurves(lightcurves)
            self._create_lightcurve_summary(lightcurves, final_candidates)
        
        return final_candidates, lightcurves

    def _combine_with_lightcurves(
    self,
    detection_tables: List[Table],
    all_epoch_detections: List[Table],
    position_match_radius: float = 2.0,
    min_n_detections: int = 3,
) -> tuple[Table, Dict]:
        """Enhanced combination that builds lightcurves during the matching process."""
        
        print(f"DEBUG: Starting _combine_with_lightcurves with {len(detection_tables)} detection tables")
        print(f"DEBUG: Parameters - position_match_radius={position_match_radius}, min_n_detections={min_n_detections}")
        
        # Load all transient candidates from individual epochs
        all_candidates = []
        candidate_sources = []  # Track which epoch each candidate came from
        
        total_loaded_candidates = 0
        
        for i, det_table in enumerate(detection_tables):
            try:
                # Use the same filename logic as above
                base_filename = self._get_base_filename(det_table, i)
                file = f"{base_filename}_transients.ecsv"
                file_path = self.transient_analyzer.data_dir / file
                
                print(f"DEBUG: Epoch {i} - Looking for file: {file}")
                
                if file_path.exists():
                    try:
                        candidates = Table.read(str(file_path))
                        
                        # Check if file is empty
                        if len(candidates) == 0:
                            print(f"DEBUG: Epoch {i} - File {file} is empty")
                            continue
                        
                        candidates['epoch_id'] = i
                        candidates['source_file'] = base_filename
                        
                        all_candidates.append(candidates)
                        candidate_sources.extend([i] * len(candidates))
                        total_loaded_candidates += len(candidates)
                        print(f"DEBUG: Epoch {i} - Loaded {len(candidates)} candidates from {file}")
                        
                    except Exception as e:
                        print(f"DEBUG: Epoch {i} - Error reading {file}: {e}")
                        continue
                else:
                    print(f"DEBUG: Epoch {i} - File not found: {file}")
                    # List what files are actually present
                    transient_files = list(self.transient_analyzer.data_dir.glob("*_transients.ecsv"))
                    if i == 0:  # Only show this once to avoid spam
                        print(f"DEBUG: Available transient files: {[f.name for f in transient_files[:5]]}...")
                    
            except Exception as e:
                print(f"DEBUG: Epoch {i} - Outer exception: {e}")
        
        print(f"DEBUG: Total candidates loaded: {total_loaded_candidates} from {len(all_candidates)} files")
        
        if not all_candidates:
            print("DEBUG: No candidate files could be loaded - returning empty results")
            return Table(), {}
        
        # Stack all candidates
        print("DEBUG: Stacking all candidates...")
        try:
            stacked_candidates = vstack(all_candidates, metadata_conflicts="silent")
            print(f"DEBUG: Stacked {len(stacked_candidates)} total candidates")
        except Exception as e:
            print(f"DEBUG: Error stacking candidates: {e}")
            return Table(), {}
        
        # Remove quality_flag column if it exists to avoid conflicts
        if "quality_flag" in stacked_candidates.colnames:
            stacked_candidates.remove_column("quality_flag")
            print("DEBUG: Removed quality_flag column")
        
        # Check for required columns
        required_cols = ["ALPHA_J2000", "DELTA_J2000", "quality_score"]
        missing_cols = [col for col in required_cols if col not in stacked_candidates.colnames]
        if missing_cols:
            print(f"DEBUG: Missing required columns: {missing_cols}")
            print(f"DEBUG: Available columns: {stacked_candidates.colnames}")
            return Table(), {}
        
        # Perform position matching using celestial coordinates
        print("DEBUG: Starting position matching...")
        try:
            ra_rad = np.radians(stacked_candidates["ALPHA_J2000"])
            dec_rad = np.radians(stacked_candidates["DELTA_J2000"])
            
            # Check for invalid coordinates
            valid_coords = np.isfinite(ra_rad) & np.isfinite(dec_rad)
            if not np.all(valid_coords):
                print(f"DEBUG: Found {np.sum(~valid_coords)} invalid coordinates, filtering them out")
                stacked_candidates = stacked_candidates[valid_coords]
                ra_rad = ra_rad[valid_coords]
                dec_rad = dec_rad[valid_coords]
                # Update candidate_sources as well
                candidate_sources = [candidate_sources[i] for i in range(len(candidate_sources)) if valid_coords[i]]
            
            if len(stacked_candidates) == 0:
                print("DEBUG: No valid coordinates remaining")
                return Table(), {}
            
            x = np.cos(dec_rad) * np.cos(ra_rad)
            y = np.cos(dec_rad) * np.sin(ra_rad)
            z = np.sin(dec_rad)
            coords = np.column_stack((x, y, z))
            
            tree = KDTree(coords)
            chord_length = 2 * np.sin(np.radians(position_match_radius / 3600) / 2)
            print(f"DEBUG: Using chord length {chord_length:.8f} for {position_match_radius} arcsec radius")
            
            groups = tree.query_radius(coords, r=chord_length)
            print(f"DEBUG: Found {len(groups)} potential groups")
            
            # Count groups by size
            group_sizes = [len(group) for group in groups]
            print(f"DEBUG: Group sizes - min: {min(group_sizes)}, max: {max(group_sizes)}, mean: {np.mean(group_sizes):.1f}")
            print(f"DEBUG: Groups with >= {min_n_detections} detections: {sum(1 for size in group_sizes if size >= min_n_detections)}")
            
        except Exception as e:
            print(f"DEBUG: Error in position matching: {e}")
            return Table(), {}
        
        # Process groups and build lightcurves
        print("DEBUG: Processing groups...")
        final_candidates = []
        lightcurves = {}
        processed = set()
        
        groups_processed = 0
        groups_with_enough_detections = 0
        groups_with_lightcurves = 0
        
        for i, group in enumerate(groups):
            if i in processed:
                continue
                
            groups_processed += 1
            
            if len(group) < min_n_detections:
                print(f"DEBUG: Group {i} has only {len(group)} detections, need {min_n_detections}")
                continue
                
            groups_with_enough_detections += 1
            
            # Get candidate data for this group
            group_candidates = stacked_candidates[group]
            group_epochs = [candidate_sources[idx] for idx in group]
            
            print(f"DEBUG: Processing group {i} with {len(group)} candidates from epochs {set(group_epochs)}")
            
            # Build comprehensive lightcurve by matching with all epoch detections
            try:
                lightcurve_data = self._build_lightcurve_for_group(
                    group_candidates, group_epochs, all_epoch_detections, position_match_radius
                )
                print(f"DEBUG: Group {i} - Built lightcurve with {len(lightcurve_data)} points")
            except Exception as e:
                print(f"DEBUG: Group {i} - Error building lightcurve: {e}")
                continue
            
            if len(lightcurve_data) >= min_n_detections:
                groups_with_lightcurves += 1
                
                # Create final candidate entry
                try:
                    best_idx = group[np.argmax(group_candidates["quality_score"])]
                    best_row = stacked_candidates[best_idx]
                    
                    # Convert Row to a new single-row Table
                    best_candidate = Table()
                    for col_name in best_row.colnames:
                        best_candidate[col_name] = [best_row[col_name]]
                    
                    # Calculate improved statistics from lightcurve
                    self._update_candidate_with_lightcurve_stats(best_candidate, lightcurve_data)
                    
                    # Generate unique ID
                    transient_id = f"transient_{best_candidate['ALPHA_J2000'][0]:.3f}_{best_candidate['DELTA_J2000'][0]:.3f}"
                    best_candidate['transient_id'] = transient_id
                    
                    final_candidates.append(best_candidate)
                    lightcurves[transient_id] = lightcurve_data
                    processed.update(group)
                    
                    print(f"DEBUG: Group {i} - Created transient {transient_id}")
                    
                except Exception as e:
                    print(f"DEBUG: Group {i} - Error creating final candidate: {e}")
                    continue
            else:
                print(f"DEBUG: Group {i} - Lightcurve has only {len(lightcurve_data)} points, need {min_n_detections}")
        
        print(f"DEBUG: Summary:")
        print(f"  - Groups processed: {groups_processed}")
        print(f"  - Groups with enough detections: {groups_with_enough_detections}")
        print(f"  - Groups with valid lightcurves: {groups_with_lightcurves}")
        print(f"  - Final candidates: {len(final_candidates)}")
        print(f"  - Lightcurves created: {len(lightcurves)}")
        
        # Convert to table
        if final_candidates:
            result_table = vstack(final_candidates)
            result_table.sort("quality_score", reverse=True)
            print(f"DEBUG: Created final table with {len(result_table)} candidates")
        else:
            result_table = Table()
            print("DEBUG: No final candidates - returning empty table")
        
        return result_table, lightcurves

    def _get_base_filename(self, det_table, epoch_index):
        """Get base filename from detection table metadata with multiple fallbacks."""
        # Try multiple metadata keys in order of preference
        filename = None
        for key in ['filename', 'FITSFILE', 'source_file', 'FILENAME']:
            if key in det_table.meta and det_table.meta[key]:
                filename = det_table.meta[key]
                print(f"DEBUG: Found filename '{filename}' using key '{key}' for epoch {epoch_index}")
                break
        
        if filename is None:
            # Final fallback
            filename = f'epoch_{epoch_index}'
            print(f"DEBUG: Using fallback filename '{filename}' for epoch {epoch_index}")
        
        # Clean up the filename
        filename = str(filename)
        
        # If it's a path, get just the basename
        filename = os.path.basename(filename)
        
        # Remove extensions properly
        for ext in ['.ecsv', '.fits', '.fit', '.csv']:
            if filename.lower().endswith(ext):
                filename = filename[:-len(ext)]
                break
        else:
            # Remove any other extension
            filename = os.path.splitext(filename)[0]
        
        # Clean up any problematic characters
        import re
        filename = re.sub(r'[^a-zA-Z0-9_\-]', '_', filename)
        filename = re.sub(r'_+', '_', filename).strip('_')
        
        # Ensure we have something valid
        if not filename or filename == 'unknown':
            filename = f'epoch_{epoch_index}'
        
        print(f"DEBUG: Final filename for epoch {epoch_index}: '{filename}'")
        return filename

    def _build_lightcurve_for_group(
    self, 
    group_candidates: Table, 
    group_epochs: List[int],
    all_epoch_detections: List[Table],
    match_radius: float
) -> Table:
        """Build a complete lightcurve by finding all detections matching the group position."""
        
        # Calculate mean position of the group
        mean_ra = np.mean(group_candidates['ALPHA_J2000'])
        mean_dec = np.mean(group_candidates['DELTA_J2000'])
        target_coord = SkyCoord(ra=mean_ra*u.deg, dec=mean_dec*u.deg)
        
        # Collect all matching detections across all epochs
        all_detections = []
        
        for epoch_id, epoch_detections in enumerate(all_epoch_detections):
            if len(epoch_detections) == 0:
                continue
                
            # Ensure coordinates are clean numpy arrays without units
            try:
                ra_values = np.array(epoch_detections['ALPHA_J2000'], dtype=float)
                dec_values = np.array(epoch_detections['DELTA_J2000'], dtype=float)
                
                # Remove any NaN or invalid values
                valid_mask = np.isfinite(ra_values) & np.isfinite(dec_values)
                if not np.any(valid_mask):
                    continue
                    
                ra_values = ra_values[valid_mask]
                dec_values = dec_values[valid_mask]
                valid_detections = epoch_detections[valid_mask]
                
                # Match positions in this epoch
                epoch_coords = SkyCoord(
                    ra=ra_values*u.deg,
                    dec=dec_values*u.deg
                )
                
                separations = target_coord.separation(epoch_coords)
                matches = separations < match_radius*u.arcsec
                
                if np.any(matches):
                    # Take the closest match if multiple
                    closest_idx = np.argmin(separations[matches])
                    matched_detection = valid_detections[matches][closest_idx]
                    all_detections.append(matched_detection)
                    
            except Exception as e:
                # If there's any issue with this epoch, skip it and continue
                print(f"Warning: Skipping epoch {epoch_id} due to coordinate issues: {e}")
                continue
        
        if all_detections:
            lightcurve = vstack(all_detections)
            # Sort by observation time
            lightcurve.sort('obs_time')
            return lightcurve
        else:
            return Table()

    def _update_candidate_with_lightcurve_stats(self, candidate: Table, lightcurve: Table):
        """Update candidate with improved statistics from full lightcurve."""
        
        # Time statistics
        time_span = (np.max(lightcurve['obs_time']) - np.min(lightcurve['obs_time'])) / 3600.0  # hours
        candidate['time_span_hours'] = time_span
        candidate['n_detections'] = len(lightcurve)
        candidate['n_epochs'] = len(np.unique(lightcurve['epoch_id']))
        
        # Position statistics
        ra_std = np.std(lightcurve['ALPHA_J2000']) * 3600  # arcsec
        dec_std = np.std(lightcurve['DELTA_J2000']) * 3600  # arcsec
        candidate['position_scatter_arcsec'] = np.sqrt(ra_std**2 + dec_std**2)
        
        # Photometric statistics
        if 'MAG_CALIB' in lightcurve.colnames:
            mags = lightcurve['MAG_CALIB']
            mag_errors = lightcurve['MAGERR_CALIB']
            
            # Weighted mean magnitude
            weights = 1.0 / mag_errors**2
            weighted_mean_mag = np.sum(mags * weights) / np.sum(weights)
            candidate['mag_weighted_mean'] = weighted_mean_mag
            
            # Magnitude variability metrics
            candidate['mag_range'] = np.max(mags) - np.min(mags)
            candidate['mag_std'] = np.std(mags)
            candidate['mag_schizo'] = np.sum(np.sqrt(np.power(np.diff(mags),2)+np.power(np.diff(lightcurve['obs_time']),2)))
            # Chi-squared test for variability
            chi2 = np.sum(((mags - weighted_mean_mag) / mag_errors)**2)
            reduced_chi2 = chi2 / (len(mags) - 1) if len(mags) > 1 else 0
            candidate['mag_chi2_reduced'] = reduced_chi2
            candidate['is_variable'] = reduced_chi2 > 2.0  # Simple variability flag
        
        # Movement analysis
        if len(lightcurve) > 2:
            times = lightcurve['obs_time']
            ra_vals = lightcurve['ALPHA_J2000']
            dec_vals = lightcurve['DELTA_J2000']
            
            # Linear fit to positions
            t_norm = (times - np.mean(times)) / 3600.0  # hours
            ra_fit = np.polyfit(t_norm, ra_vals, 1)
            dec_fit = np.polyfit(t_norm, dec_vals, 1)
            
            # Proper motion in arcsec/hour
            candidate['pm_ra_arcsec_per_hour'] = ra_fit[0] * 3600
            candidate['pm_dec_arcsec_per_hour'] = dec_fit[0] * 3600
            
            total_pm = np.sqrt(
                (ra_fit[0] * 3600 * np.cos(np.radians(np.mean(dec_vals))))**2 + 
                (dec_fit[0] * 3600)**2
            )
            candidate['total_proper_motion_arcsec_per_hour'] = total_pm

    def _analyze_and_plot_lightcurves(self, lightcurves: Dict):
        """Generate individual lightcurve plots with enhanced analysis."""
        
        for transient_id, lightcurve in lightcurves.items():
            self._plot_individual_lightcurve(transient_id, lightcurve)
            self._save_lightcurve_data(transient_id, lightcurve)

    def _plot_individual_lightcurve(self, transient_id: str, lightcurve: Table):
        """Create detailed lightcurve plot."""
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        times = lightcurve['obs_time']
        time_hours = (times - times[0]) / 3600.0  # Hours from first detection
        
        # Magnitude lightcurve
        if 'MAG_CALIB' in lightcurve.colnames:
            mags = lightcurve['MAG_CALIB']
            mag_errs = lightcurve['MAGERR_CALIB']
            ax.errorbar(time_hours, mags, yerr=mag_errs, fmt='o-', capsize=3, markersize=6)
            ax.set_ylabel('Calibrated Magnitude')
            ax.invert_yaxis()
        else:
            mags = lightcurve['MAG_ISO']
            mag_errs = lightcurve['MAGERR_ISO']
            ax.errorbar(time_hours, mags, yerr=mag_errs, fmt='o-', capsize=3, markersize=6)
            ax.set_ylabel('Instrumental Magnitude')
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
        
        # Save plot - FIX: Use Path concatenation
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
        
        # Position scatter
        ra_std = np.std(lightcurve['ALPHA_J2000']) * 3600
        dec_std = np.std(lightcurve['DELTA_J2000']) * 3600
        pos_scatter = np.sqrt(ra_std**2 + dec_std**2)
        stats.append(f"Pos scatter: {pos_scatter:.2f}\"")
        
        return '\n'.join(stats)

    def _save_lightcurve_data(self, transient_id: str, lightcurve: Table):
        """Save lightcurve data to file."""
        # Add metadata
        lightcurve.meta['transient_id'] = transient_id
        lightcurve.meta['n_detections'] = len(lightcurve)
        lightcurve.meta['n_epochs'] = len(np.unique(lightcurve['epoch_id']))
        
        # FIX: Use Path concatenation
        filename = self.lightcurve_dir / f"{transient_id}_lightcurve.ecsv"
        lightcurve.write(str(filename), format='ascii.ecsv', overwrite=True)

    def _create_lightcurve_summary(self, lightcurves: Dict, final_candidates: Table):
        """Create summary plots and analysis."""
        
        # Summary grid plot
        self._create_summary_grid_plot(lightcurves)
        
        # Variability analysis
        self._create_variability_analysis(lightcurves, final_candidates)
        
        # Create summary table
        self._create_summary_table(lightcurves, final_candidates)

    def _create_summary_grid_plot(self, lightcurves: Dict, max_plots: int = 16):
        """Create grid plot of lightcurves."""
        
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
        
        for i, (transient_id, lc) in enumerate(list(lightcurves.items())[:max_plots]):
            ax = axes_flat[i]
            
            times = lc['obs_time']
            time_hours = (times - times[0]) / 3600.0
            
            if 'MAG_CALIB' in lc.colnames:
                mags = lc['MAG_CALIB']
                mag_errs = lc['MAGERR_CALIB']
            else:
                mags = lc['MAG_ISO']
                mag_errs = lc['MAGERR_ISO']
            
            ax.errorbar(time_hours, mags, yerr=mag_errs, fmt='o-', markersize=3, capsize=2)
            ax.set_title(f"{transient_id[:15]}...\nN={len(lc)}", fontsize=8)
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=6)
        
        # Hide unused subplots
        for i in range(n_plots, len(axes_flat)):
            axes_flat[i].set_visible(False)
        
        plt.tight_layout()
        # FIX: Use Path concatenation
        save_path = self.lightcurve_dir / 'lightcurves_summary.png'
        plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
        plt.close()

    def _create_variability_analysis(self, lightcurves: Dict, final_candidates: Table):
        """Create variability analysis plots."""
        
        if len(lightcurves) == 0:
            return
            
        # Collect variability metrics
        mag_ranges = []
        mag_stds = []
        n_detections = []
        time_spans = []
        
        for lc in lightcurves.values():
            if 'MAG_CALIB' in lc.colnames:
                mags = lc['MAG_CALIB']
            else:
                mags = lc['MAG_ISO']
                
            mag_ranges.append(np.max(mags) - np.min(mags))
            mag_stds.append(np.std(mags))
            n_detections.append(len(lc))
            
            time_span = (np.max(lc['obs_time']) - np.min(lc['obs_time'])) / 3600.0
            time_spans.append(time_span)
        
        # Create variability plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Magnitude range vs number of detections
        axes[0,0].scatter(n_detections, mag_ranges)
        axes[0,0].set_xlabel('Number of detections')
        axes[0,0].set_ylabel('Magnitude range')
        axes[0,0].grid(True, alpha=0.3)
        
        # Magnitude std vs time span
        axes[0,1].scatter(time_spans, mag_stds)
        axes[0,1].set_xlabel('Time span (hours)')
        axes[0,1].set_ylabel('Magnitude std')
        axes[0,1].grid(True, alpha=0.3)
        
        # Histogram of magnitude ranges
        axes[1,0].hist(mag_ranges, bins=20, alpha=0.7)
        axes[1,0].set_xlabel('Magnitude range')
        axes[1,0].set_ylabel('Number of transients')
        axes[1,0].grid(True, alpha=0.3)
        
        # Time span distribution
        axes[1,1].hist(time_spans, bins=20, alpha=0.7)
        axes[1,1].set_xlabel('Time span (hours)')
        axes[1,1].set_ylabel('Number of transients')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.lightcurve_dir / 'variability_analysis.png'
        plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
        plt.close()

    def _create_summary_table(self, lightcurves: Dict, final_candidates: Table):
        """Create summary table of lightcurve properties."""
        
        summary_data = []
        
        for transient_id, lc in lightcurves.items():
            # Find corresponding candidate
            candidate_mask = final_candidates['transient_id'] == transient_id
            if np.any(candidate_mask):
                candidate = final_candidates[candidate_mask][0]
                
                row = {
                    'transient_id': transient_id,
                    'ra': candidate['ALPHA_J2000'],
                    'dec': candidate['DELTA_J2000'],
                    'n_detections': len(lc),
                    'n_epochs': len(np.unique(lc['epoch_id'])),
                    'time_span_hours': (np.max(lc['obs_time']) - np.min(lc['obs_time'])) / 3600.0,
                }
                
                if 'MAG_CALIB' in lc.colnames:
                    mags = lc['MAG_CALIB']
                    row.update({
                        'mag_mean': np.mean(mags),
                        'mag_range': np.max(mags) - np.min(mags),
                        'mag_std': np.std(mags)
                    })
                
                if 'quality_score' in candidate.colnames:
                    row['quality_score'] = candidate['quality_score']
                
                summary_data.append(row)
        
        if summary_data:
            summary_table = Table(summary_data)
            # FIX: Use Path concatenation
            save_path = self.lightcurve_dir / 'lightcurve_summary.ecsv'
            summary_table.write(str(save_path), format='ascii.ecsv', overwrite=True)

    def _unix_to_mjd(self, unix_time):
        """Convert Unix timestamp to Modified Julian Date."""
        try:
            t = Time(unix_time, format='unix')
            return t.mjd
        except:
            return unix_time
