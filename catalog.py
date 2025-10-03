#!/usr/bin/python3

import logging
import os
import subprocess
import tempfile
import time
import warnings
import hashlib
import pickle
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, List, Tuple, Type, TypeVar, cast
from pathlib import Path

import astropy.table
import astropy.units as u
import astropy.wcs
import numpy as np
from astropy.coordinates import SkyCoord
from sklearn.neighbors import KDTree

# Type aliases
TableType = TypeVar("TableType", bound=astropy.table.Table)
CatalogConfig = Dict[str, Any]
FilterDict = Dict[str, "CatalogFilter"]

@dataclass
class QueryParams:
    """Parameters used for catalog queries."""

    ra: Optional[float] = None
    dec: Optional[float] = None
    width: float = 0.25
    height: float = 0.25
    mlim: float = 17.0
    timeout: int = 60
    atlas_dir: str = "/home/mates/cat/atlas"

@dataclass
class CatalogOptimizationCache:
    """Cache for precomputed catalog data to avoid repeated calculations."""
    coordinates: np.ndarray
    pixel_coordinates: Dict[str, np.ndarray]  # Keyed by image identifier
    magnitudes: np.ndarray
    colors: np.ndarray
    valid_stars: np.ndarray
    kdtrees: Dict[str, KDTree]  # Cached KDTrees for each image


@dataclass
class CatalogFilter:
    """Information about a filter in a catalog."""

    name: str  # Original filter name in catalog
    effective_wl: float  # Effective wavelength in Angstroms
    system: str  # Photometric system (e.g., 'AB', 'Vega')
    error_name: Optional[str] = None  # Name of error column if available


class CatalogCache:
    """Cache management for catalog queries."""
    
    def __init__(self, cache_dir: str = "./catalog_cache"):
        """Initialize cache with specified directory."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different catalogs
        for catalog_name in ["panstarrs", "gaia", "atlas_vizier", "usno", "vsx"]:
            (self.cache_dir / catalog_name).mkdir(exist_ok=True)
    
    def _generate_cache_key(self, catalog_name: str, params: QueryParams) -> str:
        """Generate a unique cache key based on catalog and query parameters."""
        # Create a hash of the relevant parameters
        key_data = {
            'catalog': catalog_name,
            'ra': round(params.ra, 6) if params.ra else None,
            'dec': round(params.dec, 6) if params.dec else None,
            'width': round(params.width, 4),
            'height': round(params.height, 4),
            'mlim': round(params.mlim, 2)
        }
        
        # Create hash from the key data
        key_string = str(sorted(key_data.items()))
        cache_key = hashlib.md5(key_string.encode()).hexdigest()[:16]
        return cache_key
    
    def get_cache_path(self, catalog_name: str, params: QueryParams) -> Path:
        """Get the cache file path for given parameters."""
        cache_key = self._generate_cache_key(catalog_name, params)
        return self.cache_dir / catalog_name / f"{cache_key}.pkl"
    
    def is_cached(self, catalog_name: str, params: QueryParams) -> bool:
        """Check if catalog data is cached."""
        cache_path = self.get_cache_path(catalog_name, params)
        return cache_path.exists()
    
    def load_from_cache(self, catalog_name: str, params: QueryParams) -> Optional[astropy.table.Table]:
        """Load catalog data from cache."""
        cache_path = self.get_cache_path(catalog_name, params)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Verify the cached data is still valid (basic checks)
            if isinstance(cached_data, dict) and 'data' in cached_data and 'timestamp' in cached_data:
                # Check if cache is not too old (default: 30 days)
                cache_age_days = (time.time() - cached_data['timestamp']) / (24 * 3600)
                if cache_age_days < 30:
                    logging.info(f"Loading {catalog_name} from cache (age: {cache_age_days:.1f} days)")
                    return cached_data['data']
                else:
                    logging.info(f"Cache for {catalog_name} is too old ({cache_age_days:.1f} days), will refresh")
                    cache_path.unlink()  # Remove old cache
            
        except Exception as e:
            logging.info(f"Failed to load cache for {catalog_name}: {e}")
            # Remove corrupted cache file
            try:
                cache_path.unlink()
            except:
                pass
        
        return None
    
    def save_to_cache(self, catalog_name: str, params: QueryParams, data: astropy.table.Table) -> None:
        """Save catalog data to cache."""
        cache_path = self.get_cache_path(catalog_name, params)
        
        try:
            cache_data = {
                'data': data,
                'timestamp': time.time(),
                'params': asdict(params)
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            logging.info(f"Cached {catalog_name} data to {cache_path}")
            
        except Exception as e:
            logging.info(f"Failed to save cache for {catalog_name}: {e}")
    
    def clear_cache(self, catalog_name: Optional[str] = None, max_age_days: Optional[float] = None) -> None:
        """Clear cache files."""
        if catalog_name:
            cache_dirs = [self.cache_dir / catalog_name]
        else:
            cache_dirs = [d for d in self.cache_dir.iterdir() if d.is_dir()]
        
        for cache_dir in cache_dirs:
            if not cache_dir.exists():
                continue
                
            for cache_file in cache_dir.glob("*.pkl"):
                should_remove = False
                
                if max_age_days is not None:
                    try:
                        # Check file age
                        file_age = (time.time() - cache_file.stat().st_mtime) / (24 * 3600)
                        if file_age > max_age_days:
                            should_remove = True
                    except:
                        should_remove = True
                else:
                    should_remove = True
                
                if should_remove:
                    try:
                        cache_file.unlink()
                        logging.info(f"Removed cache file: {cache_file}")
                    except Exception as e:
                        logging.info(f"Failed to remove cache file {cache_file}: {e}")
    
    def get_cache_info(self) -> Any:
        """Get information about cached data."""
        info = {}
        
        for catalog_dir in self.cache_dir.iterdir():
            if not catalog_dir.is_dir():
                continue
                
            catalog_name = catalog_dir.name
            cache_files = list(catalog_dir.glob("*.pkl"))
            
            total_size = sum(f.stat().st_size for f in cache_files)
            
            info[catalog_name] = {
                'num_files': len(cache_files),
                'total_size_mb': total_size / (1024 * 1024),
                'files': []
            }
            
            for cache_file in cache_files:
                try:
                    file_age = (time.time() - cache_file.stat().st_mtime) / (24 * 3600)
                    file_size = cache_file.stat().st_size / 1024  # KB
                    
                    info[catalog_name]['files'].append({
                        'name': cache_file.name,
                        'age_days': file_age,
                        'size_kb': file_size
                    })
                except:
                    pass
        
        return info
    
    def query_vsx(self, coords: SkyCoord, radius_arcsec: float = 2.5, 
                  catalog_id: str = "B/vsx/vsx") -> Optional[astropy.table.Table]:
        """Query VSX (Variable Star Index) catalog for known variables.
        
        Args:
            coords: Sky coordinates to query around
            radius_arcsec: Search radius in arcseconds  
            catalog_id: VizieR catalog identifier for VSX
            
        Returns:
            Table with VSX matches or None if query fails
        """
        try:
            from astroquery.vizier import Vizier
            
            # Create cache key based on position and radius
            cache_params = QueryParams(
                ra=coords.ra.deg,
                dec=coords.dec.deg,
                width=radius_arcsec / 3600.0,  # Convert to degrees
                height=radius_arcsec / 3600.0
            )
            
            # Try to load from cache first
            cached_data = self.load_from_cache("vsx", cache_params)
            if cached_data is not None:
                return cached_data
            
            # Set up Vizier query
            vizier = Vizier(
                columns=["*"],  # Get all columns
                row_limit=-1,   # No row limit
                timeout=60
            )
            
            # Query VSX catalog
            logging.info(f"Querying VSX catalog {catalog_id} at {coords.ra.deg:.6f}, {coords.dec.deg:.6f} with radius {radius_arcsec}\"")
            
            result = vizier.query_region(
                coords,
                radius=radius_arcsec * u.arcsec,
                catalog=[catalog_id]
            )
            
            if not result or len(result) == 0:
                logging.debug("No VSX sources found in region")
                # Cache empty result to avoid repeated queries
                empty_table = astropy.table.Table()
                self.save_to_cache("vsx", cache_params, empty_table)
                return empty_table
            
            # Get the VSX table (should be first/only table in result)
            vsx_table = result[0]
            logging.info(f"Found {len(vsx_table)} VSX sources")
            
            # Cache the result
            self.save_to_cache("vsx", cache_params, vsx_table)
            
            return vsx_table
            
        except Exception as e:
            logging.warning(f"VSX query failed: {e}")
            return None
    
    def query_vsx_region(self, ra_deg: float, dec_deg: float, radius_arcsec: float = 2.5,
                        catalog_id: str = "B/vsx/vsx") -> Optional[astropy.table.Table]:
        """Convenience method to query VSX by RA/Dec coordinates.
        
        Args:
            ra_deg: Right ascension in degrees
            dec_deg: Declination in degrees  
            radius_arcsec: Search radius in arcseconds
            catalog_id: VizieR catalog identifier for VSX
            
        Returns:
            Table with VSX matches or None if query fails
        """
        coords = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame='icrs')
        return self.query_vsx(coords, radius_arcsec, catalog_id)


def filter_vsx_variables(candidates: astropy.table.Table, 
                        cache: CatalogCache,
                        match_radius_arcsec: float = 2.5,
                        catalog_id: str = "B/vsx/vsx") -> Tuple[astropy.table.Table, List[Dict]]:
    """Filter out candidates that match known variable stars in VSX.
    
    Args:
        candidates: Table of transient candidates with RA/Dec columns
        cache: CatalogCache instance for VSX queries
        match_radius_arcsec: Matching radius in arcseconds
        catalog_id: VizieR catalog identifier for VSX
        
    Returns:
        Tuple of (filtered_candidates, vsx_matches)
        - filtered_candidates: Table with VSX variables removed
        - vsx_matches: List of dicts with info about filtered candidates
    """
    if len(candidates) == 0:
        return candidates, []
    
    # Check if candidates have required coordinate columns
    ra_col = None
    dec_col = None
    
    # Look for common RA/Dec column names (include ALPHA_J2000/DELTA_J2000)
    for col in candidates.colnames:
        clen = col.lower()
        if clen in ['ra', 'radeg', 'ra_deg', '_ra', 'alpha_j2000']:
            ra_col = col
        elif clen in ['dec', 'decdeg', 'dec_deg', '_dec', 'delta_j2000']:
            dec_col = col
    
    if ra_col is None or dec_col is None:
        logging.warning("Could not find RA/Dec columns in candidates table for VSX filtering")
        return candidates, []
    
    logging.info(f"Starting VSX filtering for {len(candidates)} candidates")
    
    # Determine search region for VSX query
    # Get bounding box of all candidates with some padding, robust to units
    ra_values = u.Quantity(candidates[ra_col], u.deg, copy=False).to_value(u.deg)
    dec_values = u.Quantity(candidates[dec_col], u.deg, copy=False).to_value(u.deg)
    
    ra_min, ra_max = np.min(ra_values), np.max(ra_values)
    dec_min, dec_max = np.min(dec_values), np.max(dec_values)
    
    # Add padding for match radius
    padding_deg = match_radius_arcsec / 3600.0
    
    # Query VSX for the entire region
    center_ra = (ra_min + ra_max) / 2
    center_dec = (dec_min + dec_max) / 2
    
    # Calculate radius needed to cover entire region
    width_deg = (ra_max - ra_min) + 2 * padding_deg
    height_deg = (dec_max - dec_min) + 2 * padding_deg
    radius_deg = np.sqrt(width_deg**2 + height_deg**2) / 2
    radius_arcsec = radius_deg * 3600
    
    center_coords = SkyCoord(ra=center_ra * u.deg, dec=center_dec * u.deg, frame='icrs')
    
    # Query VSX
    vsx_table = cache.query_vsx(center_coords, radius_arcsec, catalog_id)
    
    if vsx_table is None or len(vsx_table) == 0:
        logging.info("No VSX sources found in candidate region")
        return candidates, []
    
    logging.info(f"Found {len(vsx_table)} VSX sources in region")
    
    # Create SkyCoord objects for matching
    candidate_coords = SkyCoord(
        ra=u.Quantity(candidates[ra_col], u.deg, copy=False),
        dec=u.Quantity(candidates[dec_col], u.deg, copy=False),
        frame='icrs'
    )
    
    # VSX coordinates (handle different possible column names)
    vsx_ra_col = None
    vsx_dec_col = None
    
    for col in vsx_table.colnames:
        clen = col.lower()
        if clen in ['ra', 'raj2000', '_raj2000', 'ra_deg', 'alpha_j2000']:
            vsx_ra_col = col
        elif clen in ['dec', 'dej2000', '_dej2000', 'dec_deg', 'delta_j2000']:
            vsx_dec_col = col
    
    if vsx_ra_col is None or vsx_dec_col is None:
        logging.warning("Could not find RA/Dec columns in VSX table")
        return candidates, []
    
    vsx_coords = SkyCoord(
        ra=u.Quantity(vsx_table[vsx_ra_col], u.deg, copy=False),
        dec=u.Quantity(vsx_table[vsx_dec_col], u.deg, copy=False),
        frame='icrs'
    )
    
    # Perform cross-matching
    match_radius = match_radius_arcsec * u.arcsec
    idx, d2d, d3d = candidate_coords.match_to_catalog_sky(vsx_coords)
    
    # Find candidates that match VSX sources within the specified radius
    matches = d2d < match_radius
    matched_candidates = candidates[matches]
    
    # Get information about the matches
    vsx_matches = []
    if np.any(matches):
        matched_idx = idx[matches]
        matched_separations = d2d[matches]
        
        for i, (cand_idx, vsx_idx, separation) in enumerate(zip(
            np.where(matches)[0], matched_idx, matched_separations
        )):
            vsx_source = vsx_table[vsx_idx]
            
            # Extract VSX information
            # Helper to safely extract scalar floats from possible Quantity/Masked
            def _to_float_deg(val):
                try:
                    # Handle Quantity with angular units
                    if hasattr(val, 'to'):
                        return float(val.to_value(u.deg))
                except Exception:
                    pass
                try:
                    return float(val)
                except Exception:
                    return np.nan

            vsx_info = {
                'candidate_index': int(cand_idx),
                'vsx_index': int(vsx_idx),
                'separation_arcsec': float(separation.arcsec),
                'vsx_name': str(vsx_source['Name']) if 'Name' in vsx_table.colnames else 'Unknown',
                'vsx_type': str(vsx_source['Type']) if 'Type' in vsx_table.colnames else 'Unknown',
                'vsx_ra': _to_float_deg(vsx_source[vsx_ra_col]),
                'vsx_dec': _to_float_deg(vsx_source[vsx_dec_col]),
                'candidate_ra': _to_float_deg(matched_candidates[ra_col][i]),
                'candidate_dec': _to_float_deg(matched_candidates[dec_col][i])
            }
            
            # Add magnitude information if available
            for mag_col in ['Vmag', 'V', 'mag']:
                if mag_col in vsx_table.colnames:
                    try:
                        vsx_info[f'vsx_{mag_col.lower()}'] = float(vsx_source[mag_col])
                        break
                    except Exception:
                        continue
            
            vsx_matches.append(vsx_info)
    
    # Filter out the matched candidates
    keep_mask = ~matches
    filtered_candidates = candidates[keep_mask]
    
    logging.info(f"VSX filtering: {len(matched_candidates)} candidates matched known variables, "
                f"{len(filtered_candidates)} candidates remain")
    
    # Log detailed removals at DEBUG level to avoid log spam
    for match in vsx_matches:
        logging.debug(
            f"Filtered VSX variable: {match['vsx_name']} (type: {match['vsx_type']}, "
            f"sep: {match['separation_arcsec']:.2f}\")"
        )
    
    return filtered_candidates, vsx_matches


class CatalogFilters:
    """Filter definitions for different catalogs."""

    PANSTARRS: FilterDict
    GAIA: FilterDict
    ATLAS: FilterDict
    USNOB: FilterDict

    # Initialize filter dictionaries as class variables
    PANSTARRS = {
        "g": CatalogFilter("gMeanPSFMag", 4810, "AB", "gMeanPSFMagErr"),
        "r": CatalogFilter("rMeanPSFMag", 6170, "AB", "rMeanPSFMagErr"),
        "i": CatalogFilter("iMeanPSFMag", 7520, "AB", "iMeanPSFMagErr"),
        "z": CatalogFilter("zMeanPSFMag", 8660, "AB", "zMeanPSFMagErr"),
        "y": CatalogFilter("yMeanPSFMag", 9620, "AB", "yMeanPSFMagErr"),
    }
    # Gaia DR3 filters
    GAIA = {
        "G": CatalogFilter("phot_g_mean_mag", 5890, "Vega", "phot_g_mean_mag_error"),
        "BP": CatalogFilter("phot_bp_mean_mag", 5050, "Vega", "phot_bp_mean_mag_error"),
        "RP": CatalogFilter("phot_rp_mean_mag", 7730, "Vega", "phot_rp_mean_mag_error"),
    }
    # ATLAS filters
    ATLAS = {
        "Sloan_g": CatalogFilter("Sloan_g", 4810, "AB"),
        "Sloan_r": CatalogFilter("Sloan_r", 6170, "AB"),
        "Sloan_i": CatalogFilter("Sloan_i", 7520, "AB"),
        "Sloan_z": CatalogFilter("Sloan_z", 8660, "AB"),
        "J": CatalogFilter("J", 12000, "AB"),
        "Johnson_B": CatalogFilter("Johnson_B", 4353, "Vega"),
        "Johnson_V": CatalogFilter("Johnson_V", 5477, "Vega"),
        "Johnson_R": CatalogFilter("Johnson_R", 6349, "Vega"),
        "Johnson_I": CatalogFilter("Johnson_I", 8797, "Vega"),
    }
    USNOB = {
        "B1": CatalogFilter("B1mag", 4500, "AB", "e_B1mag"),
        "R1": CatalogFilter("R1mag", 6400, "AB", "e_R1mag"),
        "B2": CatalogFilter("B2mag", 4500, "AB", "e_B2mag"),
        "R2": CatalogFilter("R2mag", 6400, "AB", "e_R2mag"),
        "I": CatalogFilter("Imag", 8100, "AB", "e_Imag"),
    }


class Catalog(astropy.table.Table):
    """Represents a stellar catalog with methods for retrieval and transformation.
    
    Inherits from astropy Table while providing catalog management functionality.
    """

    # Catalog identifiers
    ATLAS: str = "atlas@localhost"
    ATLAS_VIZIER: str = "atlas@vizier"
    PANSTARRS: str = "panstarrs"
    GAIA: str = "gaia"
    MAKAK: str = "makak"
    USNOB: str = "usno"

    # Class-level cache instance
    _cache = None

    KNOWN_CATALOGS: Dict[str, CatalogConfig]
    # Define available catalogs with their properties
    KNOWN_CATALOGS: Dict[str, CatalogConfig]
    KNOWN_CATALOGS = {
        ATLAS: {
            "description": "Local ATLAS catalog",
            "filters": CatalogFilters.ATLAS,
            "epoch": 2015.5,
            "local": True,
            "service": "local",
            "cacheable": False,  # Local catalogs don't need caching
            "mag_splits": [
                ("00_m_16", 0),
                ("16_m_17", 16),
                ("17_m_18", 17),
                ("18_m_19", 18),
                ("19_m_20", 19),
            ],
        },
        PANSTARRS: {
            "description": "Pan-STARRS Data Release 2",
            "filters": CatalogFilters.PANSTARRS,
            "catalog_id": "Panstarrs",
            "table": "mean",
            "release": "dr2",
            "epoch": 2015.5,
            "local": False,
            "service": "MAST",
            "cacheable": True,
            "column_mapping": {
                "raMean": "radeg",
                "decMean": "decdeg",
                "gMeanPSFMag": "gMeanPSFMag",
                "gMeanPSFMagErr": "gMeanPSFMagErr",
                "rMeanPSFMag": "rMeanPSFMag",
                "rMeanPSFMagErr": "rMeanPSFMagErr",
                "iMeanPSFMag": "iMeanPSFMag",
                "iMeanPSFMagErr": "iMeanPSFMagErr",
                "zMeanPSFMag": "zMeanPSFMag",
                "zMeanPSFMagErr": "zMeanPSFMagErr",
                "yMeanPSFMag": "yMeanPSFMag",
                "yMeanPSFMagErr": "yMeanPSFMagErr",
            },
        },
        GAIA: {
            "description": "Gaia Data Release 3",
            "filters": CatalogFilters.GAIA,
            "epoch": 2016.0,
            "local": False,
            "service": "Gaia",
            "catalog_id": "gaiadr3.gaia_source",
            "cacheable": True,
        },
        ATLAS_VIZIER: {
            "description": "ATLAS Reference Catalog 2",
            "filters": CatalogFilters.ATLAS,
            "epoch": 2015.5,
            "local": False,
            "service": "VizieR",
            "catalog_id": "J/ApJ/867/105",
            "cacheable": True,
            "column_mapping": {
                "RA_ICRS": "radeg",
                "DE_ICRS": "decdeg",
                "gmag": "Sloan_g",
                "rmag": "Sloan_r",
                "imag": "Sloan_i",
                "zmag": "Sloan_z",
                "Jmag": "J",
                "e_gmag": "Sloan_g_err",
                "e_rmag": "Sloan_r_err",
                "e_imag": "Sloan_i_err",
                "e_zmag": "Sloan_z_err",
                "e_Jmag": "J_err",
                "pmRA": "pmra",
                "pmDE": "pmdec",
            },
        },
        MAKAK: {
            "description": "Pre-filtered wide-field catalog",
            "filters": CatalogFilters.ATLAS,
            "epoch": 2015.5,
            "local": True,
            "service": "local",
            "cacheable": False,
            "filepath": "/home/mates/test/catalog.fits",
        },
        USNOB: {
            "description": "USNO-B1.0 Catalog",
            "filters": CatalogFilters.USNOB,
            "epoch": 2000.0,
            "local": False,
            "service": "VizieR",
            "catalog_id": "I/284/out",
            "cacheable": True,
            "column_mapping": {
                "RAJ2000": "radeg",
                "DEJ2000": "decdeg",
                "B1mag": "B1mag",
                "R1mag": "R1mag",
                "B2mag": "B2mag",
                "R2mag": "R2mag",
                "Imag": "Imag",
                "e_B1mag": "e_B1mag",
                "e_R1mag": "e_R1mag",
                "e_B2mag": "e_B2mag",
                "e_R2mag": "e_R2mag",
                "e_Imag": "e_Imag",
                "pmRA": "pmra",
                "pmDE": "pmdec",
            },
        },
    }

    @classmethod
    def set_cache_directory(cls, cache_dir: str) -> None:
        """Set the cache directory for all catalog instances."""
        cls._cache = CatalogCache(cache_dir)
    
    @classmethod
    def get_cache(cls) -> CatalogCache:
        """Get the cache instance, creating it if necessary."""
        if cls._cache is None:
            cls._cache = CatalogCache()
        return cls._cache
    
    @classmethod
    def clear_all_cache(cls, max_age_days: Optional[float] = None) -> None:
        """Clear all cached catalog data."""
        cache = cls.get_cache()
        cache.clear_cache(max_age_days=max_age_days)
    
    @classmethod
    def get_cache_info(cls) -> Dict[str, Any]:
        """Get information about cached catalog data."""
        cache = cls.get_cache()
        return cache.get_cache_info()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the catalog with proper handling of properties."""
        self._photometric_cache = None
        self._coordinate_cache = {}
        self._kdtree_cache = {}
        self._cache_enabled = True
        # Extract and store query parameters
        query_params: Dict[str, Any] = {}
        for param in QueryParams.__dataclass_fields__:
            if param in kwargs:
                query_params[param] = kwargs.pop(param)
        self._query_params: QueryParams = QueryParams(**query_params)

        # Store catalog name and get config
        self._catalog_name: Optional[str] = kwargs.pop("catalog", None)
        self._config: CatalogConfig = (
            self.KNOWN_CATALOGS[self._catalog_name]
            if self._catalog_name in self.KNOWN_CATALOGS
            else {}
        )

        # Initialize base Table
        if self._catalog_name:
            result = self._fetch_catalog_data()
            super().__init__(result, *args, **kwargs)
        else:
            super().__init__(*args, **kwargs)

        # Ensure catalog metadata is properly stored

        self._init_metadata()

    def _init_metadata(self) -> None:
        """Initialize or update catalog metadata."""
        if "catalog_props" not in self.meta:
            self.meta["catalog_props"] = {}

        catalog_props: Dict[str, Any] = {
            "catalog_name": self._catalog_name,
            "query_params": asdict(self._query_params) if self._query_params else None,
            "epoch": self._config.get("epoch"),
            "filters": {
                k: asdict(v) for k, v in self._config.get("filters", {}).items()
            },
            "description": self._config.get("description"),
        }

        self.meta["catalog_props"].update(catalog_props)

    @property
    def query_params(self) -> Optional[QueryParams]:
        """Get query parameters used to create this catalog."""
        params_dict = self.meta.get("catalog_props", {}).get("query_params", {})
        return QueryParams(**params_dict) if params_dict else None

    @property
    def catalog_name(self) -> str:
        """Get catalog name."""
        return str(self.meta.get("catalog_props", {}).get("catalog_name"))

    def _fetch_catalog_data(self) -> Optional[astropy.table.Table]:
        """Fetch data from the specified catalog source with caching support."""
        if self._catalog_name not in self.KNOWN_CATALOGS:
            raise ValueError(f"Unknown catalog: {self._catalog_name}")

        config = self.KNOWN_CATALOGS[self._catalog_name]
        result: Optional[astropy.table.Table] = None

        # Check if this catalog supports caching
        if config.get("cacheable", False):
            cache = self.get_cache()
            
            # Try to load from cache first
            result = cache.load_from_cache(self._catalog_name, self._query_params)
            if result is not None:
                logging.info(f"Loaded {self._catalog_name} from cache")
                # Update metadata and return
                result.meta.update(
                    {
                        "catalog": self._catalog_name,
                        "astepoch": config["epoch"],
                        "filters": list(config["filters"].keys()),
                        "cached": True
                    }
                )
                return result

        # Fetch fresh data if not cached or caching disabled
        logging.info(f"Fetching fresh data for {self._catalog_name}")
        
        if self._catalog_name == self.ATLAS:
            result = self._get_atlas_local()
        elif self._catalog_name == self.ATLAS_VIZIER:
            result = self._get_atlas_vizier()
        elif self._catalog_name == self.PANSTARRS:
            result = self._get_panstarrs_data()
        elif self._catalog_name == self.GAIA:
            result = self._get_gaia_data()
        elif self._catalog_name == self.USNOB:
            result = self._get_usnob_data()
        elif self._catalog_name == self.MAKAK:
            result = self._get_makak_data()

        if result is None:
            # Create empty catalog with proper metadata for graceful handling
            result = astropy.table.Table()
            logging.warning(f"No data found in {self._catalog_name} for this field")

        result.meta.update(
            {
                "catalog": self._catalog_name,
                "astepoch": config["epoch"],
                "filters": list(config["filters"].keys()),
                "cached": False
            }
        )
        
        # Save to cache if supported
        if config.get("cacheable", False) and len(result) > 0:
            cache = self.get_cache()
            cache.save_to_cache(self._catalog_name, self._query_params, result)
        
        return result

    # [Keep all the existing catalog-specific methods unchanged]
    def _get_atlas_local(self) -> Optional[astropy.table.Table]:
        """Get data from local ATLAS catalog."""
        config = self.KNOWN_CATALOGS[self.ATLAS]
        result: Optional[astropy.table.Table] = None

        for dirname, magspl in config["mag_splits"]:
            if self._query_params.mlim <= magspl:
                continue

            directory = os.path.join(self._query_params.atlas_dir, dirname)
            new_data = self._get_atlas_split(directory)
            if new_data is None:
                continue

            result = (
                new_data if result is None else astropy.table.vstack([result, new_data])
            )

        if result is not None and len(result) > 0:
            self._add_transformed_magnitudes(result)

        return result

    def _get_atlas_split(self, directory: str) -> Optional[astropy.table.Table]:
        """Get data from one magnitude split of ATLAS catalog."""
        with tempfile.NamedTemporaryFile(suffix=".ecsv", delete=False) as tmp:
            try:
                cmd = f'/home/mates/bin/atlas {self._query_params.ra} {self._query_params.dec} -rect {self._query_params.width},{self._query_params.height} -dir {directory} -mlim {self._query_params.mlim:.2f} -ecsv'
                logging.info(cmd)
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

                with open(tmp.name, "w") as f:
                    f.write(result.stdout)

                return astropy.table.Table.read(tmp.name, format="ascii.ecsv")

            finally:
                os.unlink(tmp.name)

    @staticmethod
    def _add_transformed_magnitudes(cat: astropy.table.Table) -> None:
        """Add transformed Johnson magnitudes."""
        gr = cat["Sloan_g"] - cat["Sloan_r"]
        ri = cat["Sloan_r"] - cat["Sloan_i"]
        iz = cat["Sloan_i"] - cat["Sloan_z"]

        cat["Johnson_B"] = (
            cat["Sloan_r"]
            + 1.490989 * gr
            + 0.125787 * gr * gr
            - 0.022359 * gr * gr * gr
            + 0.186304
        )
        cat["Johnson_V"] = cat["Sloan_r"] + 0.510236 * gr - 0.0337082
        cat["Johnson_R"] = (
            cat["Sloan_r"] - 0.197420 * ri - 0.083113 * ri * ri - 0.179943
        )
        cat["Johnson_I"] = cat["Sloan_r"] - 0.897087 * ri - 0.575316 * iz - 0.423971

    def _get_atlas_vizier(self) -> Optional[astropy.table.Table]:
        """Get ATLAS RefCat2 data from VizieR with updated column mapping."""
        from astroquery.vizier import Vizier

        column_mapping = self.KNOWN_CATALOGS[self.ATLAS_VIZIER]["column_mapping"]
        vizier = Vizier(
            columns=list(column_mapping.keys()),
            column_filters={
                "rmag": f"<{self._query_params.mlim}"
            },
            row_limit=-1,
        )

        coords = SkyCoord(
            ra=self._query_params.ra * u.deg,
            dec=self._query_params.dec * u.deg,
            frame="icrs",
        )

        result = vizier.query_region(
            coords,
            width=self._query_params.width * u.deg,
            height=self._query_params.height * u.deg,
            catalog=self.KNOWN_CATALOGS[self.ATLAS_VIZIER]["catalog_id"],
        )

        if not result or len(result) == 0:
            return None

        atlas = result[0]
        cat = astropy.table.Table(result)

        our_columns = set(column_mapping.values())
        for col in our_columns:
            cat[col] = np.zeros(len(atlas), dtype=np.float64)

        for vizier_name, our_name in column_mapping.items():
            if vizier_name in atlas.columns:
                if vizier_name in ["pmRA", "pmDE"]:
                    cat[our_name] = atlas[vizier_name] / (3.6e6)
                else:
                    cat[our_name] = atlas[vizier_name]

        self._add_transformed_magnitudes(cat)
        return cat

    def _get_panstarrs_data(self) -> Optional[astropy.table.Table]:
        """Get PanSTARRS DR2 data."""
        from astroquery.mast import Catalogs

        config = self.KNOWN_CATALOGS[self.PANSTARRS]
        radius = (
            np.sqrt(self._query_params.width ** 2 + self._query_params.height ** 2) / 2
        )
        coords = SkyCoord(
            ra=self._query_params.ra * u.deg,
            dec=self._query_params.dec * u.deg,
            frame="icrs",
        )

        constraints = {
            "nDetections.gt": 4,
            "rMeanPSFMag.lt": self._query_params.mlim,
            "qualityFlag.lt": 128,
        }

        ps1 = Catalogs.query_region(
            coords,
            catalog=config["catalog_id"],
            radius=radius * u.deg,
            data_release="dr2",
            table=config["table"],
            **constraints,
        )

        if len(ps1) == 0:
            return None

        result = astropy.table.Table()

        for ps1_name, our_name in config["column_mapping"].items():
            if ps1_name in ps1.columns:
                result[our_name] = ps1[ps1_name].astype(np.float64)

        result["pmra"] = np.zeros(len(ps1), dtype=np.float64)
        result["pmdec"] = np.zeros(len(ps1), dtype=np.float64)

        return result

    def _get_gaia_data(self) -> Optional[astropy.table.Table]:
        """Get Gaia DR3 data."""
        try:
            from astroquery.gaia import Gaia

            config = self.KNOWN_CATALOGS[self.GAIA]
            query = f"""
            SELECT
                source_id, ra, dec, pmra, pmdec,
                phot_g_mean_mag, phot_g_mean_flux_over_error,
                phot_bp_mean_mag, phot_bp_mean_flux_over_error,
                phot_rp_mean_mag, phot_rp_mean_flux_over_error
            FROM {config['catalog_id']}
            WHERE 1=CONTAINS(
                POINT('ICRS', ra, dec),
                BOX('ICRS', {self._query_params.ra}, {self._query_params.dec}, {2*self._query_params.width}, {2*self._query_params.height}))
                AND phot_g_mean_mag < {self._query_params.mlim}
                AND ruwe < 1.4
                AND visibility_periods_used >= 8
                -- Ensure we only get complete photometric data
                AND phot_g_mean_mag IS NOT NULL
                AND phot_bp_mean_mag IS NOT NULL
                AND phot_rp_mean_mag IS NOT NULL
                AND phot_g_mean_flux_over_error > 0
                AND phot_bp_mean_flux_over_error > 0
                AND phot_rp_mean_flux_over_error > 0
            """

            job = Gaia.launch_job_async(query)
            gaia_cat = job.get_results()

            if len(gaia_cat) == 0:
                logging.warning("No Gaia data found")
                return None

            result = astropy.table.Table()

            result["radeg"] = gaia_cat["ra"]
            result["decdeg"] = gaia_cat["dec"]
            result["pmra"] = gaia_cat["pmra"] / (3.6e6)
            result["pmdec"] = gaia_cat["pmdec"] / (3.6e6)

            for filter_name, filter_info in config["filters"].items():
                result[filter_info.name] = gaia_cat[filter_info.name]
                if filter_info.error_name:
                    flux_over_error = gaia_cat[
                        filter_info.name.replace("mag", "flux_over_error")
                    ]
                    result[filter_info.error_name] = 2.5 / (
                        flux_over_error * np.log(10)
                    )

            return result

        except Exception as e:
            raise ValueError(f"Gaia query failed: {str(e)}") from e

    def _get_usnob_data(self) -> Optional[astropy.table.Table]:
        """Get USNO-B1.0 data from VizieR."""
        try:
            from astroquery.vizier import Vizier

            config = self.KNOWN_CATALOGS[self.USNOB]
            column_mapping = config["column_mapping"]

            vizier = Vizier(
                columns=list(column_mapping.keys()),
                column_filters={
                    "R1mag": f"<{self._query_params.mlim}"
                },
                row_limit=-1,
            )

            coords = SkyCoord(
                ra=self._query_params.ra * u.deg,
                dec=self._query_params.dec * u.deg,
                frame="icrs",
            )

            result = vizier.query_region(
                coords,
                width=2*self._query_params.width * u.deg,
                height=2*self._query_params.height * u.deg,
                catalog=config["catalog_id"],
            )

            if not result or len(result) == 0:
                logging.info("No USNO-B data found")
                return None

            usnob = result[0]
            cat = astropy.table.Table()

            our_columns = set(column_mapping.values())
            for col in our_columns:
                cat[col] = np.zeros(len(usnob), dtype=np.float64)

            for vizier_name, our_name in column_mapping.items():
                if vizier_name in usnob.columns:
                    if vizier_name in ["pmRA", "pmDE"]:
                        cat[our_name] = usnob[vizier_name] / (3.6e6)
                    else:
                        cat[our_name] = usnob[vizier_name]

            for band in ["B1", "R1", "B2", "R2", "I"]:
                mag_col = f"{band}mag"
                err_col = f"e_{band}mag"
                if mag_col in cat.columns:
                    if err_col not in cat.columns or np.all(cat[err_col] == 0):
                        cat[err_col] = np.where(
                            cat[mag_col] < 19,
                            0.1,
                            0.2,
                        )

            return cat

        except Exception as e:
            raise ValueError(f"USNO-B query failed: {str(e)}") from e

    def _get_makak_data(self) -> Optional[astropy.table.Table]:
        """Get data from pre-filtered MAKAK catalog."""
        try:
            from astroquery.vizier import Vizier

            cat = astropy.table.Table.read(config["filepath"])
            if self._query_params.ra is None or self._query_params.dec is None:
                raise ValueError("RA and DEC are required for MAKAK catalog access")
            
            ctr = SkyCoord(
                self._query_params.ra * u.deg,
                self._query_params.dec * u.deg,
                frame="fk5",
            )
            corner = SkyCoord(
                (self._query_params.ra + self._query_params.width) * u.deg,
                (self._query_params.dec + self._query_params.height) * u.deg,
                frame="fk5",
            )
            radius = corner.separation(ctr) / 2

            cat_coords = SkyCoord(
                cat["radeg"] * u.deg, cat["decdeg"] * u.deg, frame="fk5"
            )
            within_field = cat_coords.separation(ctr) < radius
            cat = cat[within_field]

            # Query VizieR (use 2x width/height for SDSS like catalog+sdss.py does)
            result = vizier.query_region(
                coords,
                width=2*self._query_params.width * u.deg,
                height=2*self._query_params.height * u.deg,
                catalog=config['catalog_id']
            )

            if not result or len(result) == 0:
                logging.warning("No SDSS data found")
                return None

            if "pmra" not in cat.columns:
                cat["pmra"] = np.zeros(len(cat), dtype=np.float64)
            if "pmdec" not in cat.columns:
                cat["pmdec"] = np.zeros(len(cat), dtype=np.float64)

            return cat

        except Exception as e:
            raise ValueError(f"MAKAK catalog access failed: {str(e)}") from e

    @classmethod
    def from_file(cls: Type[TableType], filename: str) -> TableType:
        """Create catalog instance from a local file with proper metadata handling."""
        try:
            data = astropy.table.Table.read(filename)
            obj = cls(data.as_array())
            obj.meta.update(data.meta)

            if "catalog_props" not in obj.meta:
                obj.meta["catalog_props"] = {
                    "catalog_name": "local",
                    "description": f"Local catalog from {filename}",
                    "epoch": None,
                    "filters": {},
                }

            return cast(TableType, obj)

        except Exception as e:
            raise ValueError(f"Failed to read catalog from {filename}: {str(e)}") from e

    @property
    def description(self) -> str:
        """Get catalog description."""
        return str(
            self.meta.get("catalog_props", {}).get("description", "Unknown catalog")
        )

    @property
    def filters(self) -> Dict[str, CatalogFilter]:
        """Get available filters."""
        filters_dict = self.meta.get("catalog_props", {}).get("filters", {})
        return {k: CatalogFilter(**v) for k, v in filters_dict.items()}

    @property
    def epoch(self) -> float:
        """Get catalog epoch."""
        return float(self.meta.get("catalog_props", {}).get("epoch"))

    def __array_finalize__(self, obj: Optional[astropy.table.Table]) -> None:
        """Ensure proper handling of metadata during numpy operations."""
        super().__array_finalize__(obj)
        if obj is None:
            return

        if hasattr(obj, "meta") and "catalog_props" in obj.meta:
            if not hasattr(self, "meta"):
                self.meta = {}
            self.meta["catalog_props"] = obj.meta["catalog_props"].copy()

    def copy(self, copy_data: bool = True) -> astropy.table.Table:
        """Create a copy ensuring catalog properties are preserved."""
        new_cat = super().copy(copy_data=copy_data)
        if "catalog_props" in self.meta:
            new_cat.meta["catalog_props"] = self.meta["catalog_props"].copy()
        return new_cat

# Utility functions for cache management
def setup_catalog_cache(cache_dir: str = "./catalog_cache") -> None:
    """Setup catalog caching with specified directory."""
    Catalog.set_cache_directory(cache_dir)
    logging.info(f"Catalog cache set to: {cache_dir}")


def print_cache_info() -> None:
    """Print information about cached catalog data."""
    info = Catalog.get_cache_info()
    
    logging.info("\n=== Catalog Cache Information ===")
    total_size = 0
    total_files = 0
    
    for catalog_name, cat_info in info.items():
        logging.info(f"\n{catalog_name.upper()}:")
        logging.info(f"  Files: {cat_info['num_files']}")
        logging.info(f"  Total size: {cat_info['total_size_mb']:.1f} MB")
        
        total_size += cat_info['total_size_mb']
        total_files += cat_info['num_files']
        
        if cat_info['files']:
            logging.info("  Recent files:")
            # Show 3 most recent files
            recent_files = sorted(cat_info['files'], key=lambda x: x['age_days'])[:3]
            for file_info in recent_files:
                logging.info(f"    {file_info['name']} ({file_info['age_days']:.1f} days, {file_info['size_kb']:.0f} KB)")
    
    logging.info(f"\nTotal: {total_files} files, {total_size:.1f} MB")


def clear_old_cache(max_age_days: float = 30.0) -> None:
    """Clear cache files older than specified age."""
    logging.info(f"Clearing cache files older than {max_age_days} days...")
    Catalog.clear_all_cache(max_age_days=max_age_days)
    logging.info("Cache cleanup completed.")


def clear_all_cache() -> None:
    """Clear all cached catalog data."""
    logging.info("Clearing all cached catalog data...")
    Catalog.clear_all_cache()
    logging.info("All cache cleared.")


# Example usage functions
def main():
    """Example of how to use the cached catalog system."""
    
    # Setup cache directory
    setup_catalog_cache("./my_catalog_cache")
    
    # Define query parameters
    params = QueryParams(
        ra=150.0,
        dec=2.0,
        width=0.5,
        height=0.5,
        mlim=18.0
    )
    
    logging.info("First query (will download and cache):")
    cat1 = Catalog(catalog="gaia", **params.__dict__)
    logging.info(f"Got {len(cat1)} sources from Gaia")
    
    logging.info("\nSecond query (should load from cache):")
    cat2 = Catalog(catalog="gaia", **params.__dict__)
    logging.info(f"Got {len(cat2)} sources from Gaia")
    
    # Show cache info
    print_cache_info()
    
    # Example with PanSTARRS
    logging.info("\nQuerying PanSTARRS (will cache if successful):")
    try:
        cat_ps = Catalog(catalog="panstarrs", **params.__dict__)
        logging.info(f"Got {len(cat_ps)} sources from PanSTARRS")
    except Exception as e:
        logging.info(f"PanSTARRS query failed: {e}")
    
    # Final cache info
    print_cache_info()


if __name__ == "__main__":
    main()
