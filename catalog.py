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

    # [Rest of the methods remain unchanged...]
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
    
    def _legacy_get_transient_candidates(self, det: astropy.table.Table, idlimit: float = 5.0) -> astropy.table.Table:
        """Legacy transient detection method for fallback."""
        try:
            self._validate_detection_table(det)

            cat_xy = self._transform_catalog_to_pixel(det)
            if len(cat_xy) < 1:
                warnings.warn("No valid catalog sources in the field")
                return det

            det_xy = np.array([det["X_IMAGE"], det["Y_IMAGE"]]).T

            tree = KDTree(cat_xy)

            indices, distances = tree.query_radius(
                det_xy, r=idlimit, return_distance=True
            )
            
            transient_mask = np.array([len(idx) == 0 for idx in indices])

            transients = det[transient_mask].copy()

            return transients

        except Exception as e:
            raise ValueError(f"Transient detection failed: {str(e)}") from e

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
        
        if cat_xy is None:
            import warnings
            warnings.warn("Could not transform catalog coordinates, falling back to legacy method")
            return self._legacy_get_transient_candidates(detections, idlimit)
        
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
