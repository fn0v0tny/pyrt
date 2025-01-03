import warnings
from typing import Dict, List, Optional

import numpy as np
from astropy.table import Table, vstack
from sklearn.neighbors import KDTree

from catalog import Catalog, QueryParams


class TransientAnalyzer:
    """Combined system for transient detection and feature extraction."""

    def __init__(self) -> None:
        # Detection features we want to extract if available
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
    ) -> Dict[str, Table]:
        """Find and analyze transient candidates using multiple catalogs.
        
        Args:
            detections: Detection table
            catalogs: List of catalog names to check
            params: Query parameters for catalog access
            idlimit: Identification radius for transient detection
            radius_check: Radius for checking catalog properties
            
        Returns:
            Dict mapping catalog names to tables of analyzed candidates
        """
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

        # Photometric features
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
        cat_count = len(set(all_candidates[idx]["reference_catalog"] for idx in group))

        if cat_count >= min_catalogs:
            # Take the one with highest quality score from the group
            group_qualities = all_candidates["quality_score"][group]
            best_idx = group[np.argmax(group_qualities)]
            reliable.append(all_candidates[best_idx])
            processed.update(group)

    if reliable:
        return vstack(reliable)
    return Table()
