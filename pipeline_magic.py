#!/usr/bin/python3

import sys
import shutil
import json
import subprocess
import time
import logging
import hashlib
import yaml
from catalog import QueryParams, setup_catalog_cache
from transients import *
from transient_analyser import *
from extraction_manager import ImageExtractionManager
from config_trans import PipelineConfig
import os
import warnings
from pathlib import Path
from datetime import datetime

# Reduce noisy FITS header warnings from astropy (HIERARCH cards etc.)
try:
    from astropy.io.fits.verify import VerifyWarning
    warnings.filterwarnings("ignore", category=VerifyWarning)
except Exception:
    pass
try:
    from astropy.utils.exceptions import AstropyWarning
    warnings.filterwarnings("ignore", category=AstropyWarning)
except Exception:
    pass

def extract_observation_id(ecsv_file_path):
    """Extract observation ID from ECSV file metadata or filename."""
    try:
        # Try to get observation ID from file metadata first
        ecsv_data = open_ecsv_file(ecsv_file_path, verbose=False)
        if ecsv_data and ecsv_data.meta:
            # Check common observation ID fields
            for field in ['OBSID', 'OBS_ID', 'OBSERVATION_ID', 'FIELD_ID']:
                if field in ecsv_data.meta:
                    obs_id = str(ecsv_data.meta[field])
                    # Clean observation ID - remove decimal part
                    return clean_observation_id(obs_id)
        
        # Fallback: extract from filename (assuming pattern like obs_12345_...)
        filename = Path(ecsv_file_path).stem
        parts = filename.split('_')
        for i, part in enumerate(parts):
            if part.lower() in ['obs', 'obsid', 'field'] and i + 1 < len(parts):
                obs_id = parts[i + 1]
                # Clean observation ID - remove decimal part
                return clean_observation_id(obs_id)
        
        # Last resort: use filename without extension
        return clean_observation_id(filename)
        
    except Exception as e:
        logging.warning(f"Could not extract observation ID from {ecsv_file_path}: {e}")
        return clean_observation_id(Path(ecsv_file_path).stem)

def clean_observation_id(obs_id):
    """Clean observation ID by removing decimal parts and invalid characters."""
    obs_id = str(obs_id).strip()
    
    # Remove decimal part (e.g., 94249.01 -> 94249)
    if '.' in obs_id:
        obs_id = obs_id.split('.')[0]
    
    # Remove any other problematic characters and keep only alphanumeric and underscores
    import re
    obs_id = re.sub(r'[^a-zA-Z0-9_]', '_', obs_id)
    
    # Remove multiple consecutive underscores
    obs_id = re.sub(r'_+', '_', obs_id)
    
    # Remove leading/trailing underscores
    obs_id = obs_id.strip('_')
    
    # Ensure we have something valid
    if not obs_id:
        obs_id = "unknown"
    
    return obs_id

def setup_observation_directory(base_dir, observation_id):
    """Create and return observation-specific directory with file locking."""
    obs_dir = Path(base_dir) / f"obs_{observation_id}"
    
    # Use a lock file to prevent race conditions when multiple processes
    # try to create the same directory simultaneously
    lock_file = Path(base_dir) / f".{observation_id}.lock"
    
    try:
        # Try to create directory (race condition safe)
        obs_dir.mkdir(exist_ok=True)
        return obs_dir
    except Exception as e:
        logging.warning(f"Could not create observation directory {obs_dir}: {e}")
        # Fallback to base directory if obs directory creation fails
        return Path(base_dir)

def get_existing_detection_tables(obs_dir):
    """Load existing detection tables from observation directory."""
    detection_tables = []
    metadata_file = obs_dir / "detection_metadata.json"
    
    # Load metadata about processed files
    processed_files = set()
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                processed_files = set(metadata.get('processed_files', []))
        except Exception as e:
            logging.warning(f"Could not load metadata: {e}")
    
    # Load existing detection tables
    for ecsv_file in obs_dir.glob("*.ecsv"):
        if ecsv_file.name in processed_files:
            try:
                detection_data = open_ecsv_file(str(ecsv_file))
                if detection_data is not None and detection_data.meta is not None:
                    detection_tables.append(detection_data)
                    logging.info(f"Loaded existing detection table: {ecsv_file.name}")
            except Exception as e:
                logging.warning(f"Could not load {ecsv_file}: {e}")
    
    return detection_tables, processed_files

def update_metadata(obs_dir, processed_files, new_file):
    """Update metadata file with newly processed file (thread-safe)."""
    import fcntl
    import tempfile
    
    processed_files.add(new_file)
    metadata_file = obs_dir / "detection_metadata.json"
    
    metadata = {
        'processed_files': list(processed_files),
        'last_updated': datetime.now().isoformat(),
        'total_files': len(processed_files),
        'observation_id': obs_dir.name.replace('obs_', ''),
        'process_id': os.getpid()
    }
    
    try:
        # Use atomic write with temporary file to prevent corruption
        temp_file = metadata_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            # Try to get exclusive lock (non-blocking)
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                json.dump(metadata, f, indent=2)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except BlockingIOError:
                # If we can't get the lock, another process is updating
                logging.info(f"Another process is updating metadata, skipping...")
                return
        
        # Atomic move
        temp_file.replace(metadata_file)
        
    except Exception as e:
        logging.warning(f"Could not save metadata: {e}")
        # Clean up temp file if it exists
        if temp_file.exists():
            temp_file.unlink()

def check_if_already_processed(obs_dir, ecsv_basename):
    """Check if this specific file has already been processed (thread-safe)."""
    import fcntl
    
    metadata_file = obs_dir / "detection_metadata.json"
    
    if not metadata_file.exists():
        return False
    
    try:
        with open(metadata_file, 'r') as f:
            # Try to get shared lock for reading
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
                metadata = json.load(f)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                
                processed_files = set(metadata.get('processed_files', []))
                return ecsv_basename in processed_files
            except BlockingIOError:
                # If we can't get the lock, assume not processed to be safe
                logging.info(f"INFO: Could not read metadata (locked), assuming not processed")
                return False
    except Exception as e:
        logging.info(f"WARNING: Could not read metadata: {e}")
        return False

def _compute_file_md5(path, block_size=65536):
    """Compute MD5 checksum of a file, returns hex string or None if missing."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    md5 = hashlib.md5()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def _load_site_state(website_dir):
    state_file = Path(website_dir) / ".site_state.json"
    if not state_file.exists():
        return None
    try:
        with open(state_file, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def _save_site_state(website_dir, state):
    state_file = Path(website_dir) / ".site_state.json"
    try:
        Path(website_dir).mkdir(parents=True, exist_ok=True)
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logging.info(f"WARNING: Could not save site state: {e}")


def generate_frontend(obs_dir, observation_id, config):
    """Generate frontend using the integrated FrontendGenerator, gated by candidates.tbl changes."""
    try:
        from frontend_generator import FrontendGenerator
        
        # Create frontend generator
        base_public_dir = config.base_public_dir or Path.home() / "public_html"
        website_dir = Path(base_public_dir) / f"obs_{observation_id}"
        candidates_file = Path(obs_dir) / "candidates.tbl"

        # Gate by checksum: skip if unchanged and site exists
        current_hash = _compute_file_md5(candidates_file)
        prev_state = _load_site_state(website_dir)
        index_exists = (website_dir / "index.html").exists()
        if index_exists and prev_state and prev_state.get("candidates_md5") == current_hash and current_hash is not None:
            logging.info(f"Website up-to-date for observation {observation_id}, skipping generation")
            return True

        logging.info(f"Generating website for observation {observation_id}...")
        frontend_gen = FrontendGenerator(
            observation_id=observation_id,
            data_dir=obs_dir,
            base_public_dir=base_public_dir,
            config=config.frontend
        )
        
        # Generate the complete website
        success = frontend_gen.generate_complete_website()

        # Persist new state on success
        if success and current_hash is not None:
            _save_site_state(website_dir, {
                "candidates_md5": current_hash,
                "updated": datetime.now().isoformat()
            })
        
        return success
        
            
    except ImportError as e:
        logging.info(f"ERROR: Could not import FrontendGenerator: {e}")
        logging.info("Falling back to old method...")
        # Fallback to old method if needed
        return generate_frontend_old(obs_dir)
    except Exception as e:
        logging.info(f"ERROR: Frontend generation failed: {e}")
        return False

def generate_frontend_old(obs_dir):
    """Fallback to old frontend generation method."""
    try:
        frontend_script = Path("/home/fnovotny/bin/gen_frontend.py")
        
        if frontend_script.exists():
            logging.info(f"Generating frontend for observation in {obs_dir}")
            result = subprocess.run([
                str(frontend_script), str(obs_dir)
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                logging.info("Frontend generated successfully")
                if result.stdout:
                    logging.info(f"Frontend output: {result.stdout.strip()}")
                return True
            else:
                logging.info(f"Frontend generation failed with exit code {result.returncode}")
                if result.stderr:
                    logging.info(f"Frontend error: {result.stderr}")
                return False
        else:
            logging.info(f"Frontend script not found at {frontend_script}")
            logging.info("Skipping frontend generation")
            return False
            
    except subprocess.TimeoutExpired:
        logging.info("WARNING: Frontend generation timed out")
        return False
    except Exception as e:
        logging.info(f"WARNING: Frontend generation failed: {e}")
        return False

def process_observation(obs_dir, detection_tables, first_det, config):
    """Run the full transient analysis pipeline."""
    logging.info("Running transient analysis...")
    
    # Setup image manager
    image_manager = ImageExtractionManager(detection_tables)
    ra, dec = image_manager.field_center
    logging.info(f"Field center: RA={ra:.6f}, DEC={dec:.6f}")
    
    # Setup query parameters
    query_params = QueryParams(**{
        "ra": ra,
        "dec": dec,
        "width": 1.2 * first_det.meta["FIELD"],
        "height": 1.2 * first_det.meta["FIELD"],
        "mlim": 20,
    })
    
    logging.info(f"Query parameters: RA={ra:.6f}, DEC={dec:.6f}, "
          f"width={1.2 * first_det.meta['FIELD']:.3f}, "
          f"height={1.2 * first_det.meta['FIELD']:.3f}")
    
    # Run transient analysis
    transient_analyzer = OptimizedTransientAnalyzer(data_dir=obs_dir, config=config)
    multi_analyzer = OptimizedMultiDetectionAnalyzer(transient_analyzer, lightcurve_dir=obs_dir, config=config)
    
    logging.info("Processing detection tables...")
    # FIX: Unpack the tuple returned by the method
    reliable_candidates, lightcurves = multi_analyzer.process_detection_tables_with_lightcurves(
        detection_tables=detection_tables,
        catalogs=["atlas@localhost", "gaia", "usno"],
        params=query_params,
        idlimit=config.detection.idlimit_px,
        radius_check=config.detection.radius_check,
        filter_pattern=config.detection.filter_pattern,
        min_n_detections=config.detection.min_n_detections,
        min_catalogs=config.detection.min_catalogs,
        min_quality=config.detection.min_quality,
        position_match_radius=config.detection.position_match_radius_arcsec,
    )
    
    logging.info(f"Found {len(reliable_candidates)} reliable candidates")
    if len(reliable_candidates) > 0:
        logging.info(reliable_candidates)
    
    # Save results
    candidates_file = obs_dir / "candidates.tbl"
    reliable_candidates.write(
        str(candidates_file), format="ascii.ipac", overwrite=True
    )
    logging.info(f"Results saved to {candidates_file}")
    
    # Also save lightcurve information if available
    if lightcurves:
        logging.info(f"Generated {len(lightcurves)} lightcurves")
        lightcurve_summary_file = obs_dir / "lightcurve_summary.json"
        
        # Create a simple summary of lightcurves
        lightcurve_summary = {}
        for transient_id, lc_data in lightcurves.items():
            lightcurve_summary[transient_id] = {
                'n_detections': len(lc_data),
                'n_epochs': len(np.unique(lc_data['epoch_id'])) if 'epoch_id' in lc_data.colnames else 1,
                'time_span_hours': float((np.max(lc_data['obs_time']) - np.min(lc_data['obs_time'])) / 3600.0) if 'obs_time' in lc_data.colnames else 0.0
            }
        
        try:
            import json
            with open(lightcurve_summary_file, 'w') as f:
                json.dump(lightcurve_summary, f, indent=2)
            logging.info(f"Lightcurve summary saved to {lightcurve_summary_file}")
        except Exception as e:
            logging.info(f"WARNING: Could not save lightcurve summary: {e}")
    
    return reliable_candidates

def should_run_analysis(obs_dir, new_detection_added):
    """Determine if analysis should be run (with coordination between parallel processes)."""
    candidates_file = obs_dir / "candidates.tbl"
    analysis_lock = obs_dir / ".analysis.lock"
    
    logging.info(f"Checking analysis conditions:")
    logging.info(f"  - New detection added: {new_detection_added}")
    logging.info(f"  - Candidates file exists: {candidates_file.exists()}")
    logging.info(f"  - Analysis lock exists: {analysis_lock.exists()}")
    
    # If we added new data, we definitely need analysis
    if new_detection_added:
        logging.info("  → Decision: Need analysis (new detection data)")
        return True, "New detection data added"
    
    # If no results exist, we need analysis
    if not candidates_file.exists():
        logging.info("  → Decision: Need analysis (no existing results)")
        return True, "No existing results found"
    
    # Check if another process is already running analysis
    if analysis_lock.exists():
        # Check if the lock is stale (older than 15 minutes)
        try:
            lock_age = time.time() - analysis_lock.stat().st_mtime
            logging.info(f"  - Lock age: {lock_age/60:.1f} minutes")
            if lock_age > 900:  # 15 minutes
                logging.info(f"  → Decision: Removing stale lock and running analysis")
                analysis_lock.unlink()
                return True, "Stale lock removed, rerunning analysis"
            else:
                logging.info(f"  → Decision: Skip analysis (another process running)")
                return False, f"Analysis already running (lock age: {lock_age/60:.1f} minutes)"
        except Exception as e:
            logging.info(f"  - Error checking lock: {e}")
            logging.info(f"  → Decision: Skip analysis (lock check failed)")
            return False, "Analysis lock present (cannot check age)"
    
    logging.info("  → Decision: Skip analysis (results exist, no new data)")
    return False, "Results exist and no new data"

def create_analysis_lock(obs_dir):
    """Create analysis lock file."""
    analysis_lock = obs_dir / ".analysis.lock"
    try:
        logging.info(f"Creating analysis lock: {analysis_lock}")
        with open(analysis_lock, 'w') as f:
            f.write(f"PID: {os.getpid()}\nStarted: {datetime.now().isoformat()}\n")
        logging.info(f"Analysis lock created successfully")
        return analysis_lock
    except Exception as e:
        logging.info(f"WARNING: Could not create analysis lock: {e}")
        return None

def remove_analysis_lock(analysis_lock):
    """Remove analysis lock file."""
    if analysis_lock and analysis_lock.exists():
        try:
            analysis_lock.unlink()
            logging.info("Analysis lock removed")
        except Exception as e:
            logging.info(f"WARNING: Could not remove analysis lock: {e}")

def load_config_with_yaml_support(config_file):
    """Load configuration with YAML support."""
    config_path = Path(config_file)
    
    # Check if it's a YAML file
    if config_path.suffix.lower() in ['.yaml', '.yml']:
        try:
            with open(config_path, 'r') as f:
                yaml_data = yaml.safe_load(f)
            
            # Create PipelineConfig from YAML data
            return PipelineConfig.from_dict(yaml_data)
        except Exception as e:
            logging.error(f"Failed to load YAML config from {config_file}: {e}")
            raise
    else:
        # Use existing from_file method for non-YAML files
        return PipelineConfig.from_file(config_file)

def setup_pipeline_logging(config, observation_id):
    """Setup comprehensive logging for the pipeline."""
    # Create logs directory
    log_dir = Path(config.base_data_dir) / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Setup file handlers
    log_file = log_dir / f"pipeline_{observation_id}.log"
    debug_log_file = log_dir / f"pipeline_{observation_id}_debug.log"
    
    # Get root logger and clear existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers = []
    
    # Set logging level
    if config.logging.level == "DEBUG":
        root_logger.setLevel(logging.DEBUG)
    else:
        root_logger.setLevel(logging.INFO)
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s')
    
    # File handler for INFO+ messages
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # File handler for DEBUG+ messages
    debug_handler = logging.FileHandler(debug_log_file)
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(formatter)
    root_logger.addHandler(debug_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    return logging.getLogger('pipeline_magic')

def main():
    if len(sys.argv) < 3:
        logging.error("Usage: pipeline_magic.py <ecsv_file> <fits_file> [base_data_dir] [--generate-frontend] [--config=<config_file>] [--output-dir=<path>] [--debug]")
        logging.error("  --generate-frontend: Optional flag to generate website after analysis")
        logging.error("  --config=<file>: Optional config file path (supports .yaml/.yml)")
        logging.error("  --output-dir=<path>: Override base data directory")
        logging.error("  --debug: Enable debug logging")
        sys.exit(1)

    # Parse command line arguments
    ecsv_file = sys.argv[1]
    fits_file = sys.argv[2]
    
    # Parse optional arguments
    generate_frontend_flag = "--generate-frontend" in sys.argv
    debug_flag = "--debug" in sys.argv
    config_file = None
    base_data_dir = None
    output_dir = None
    
    for arg in sys.argv[3:]:
        if arg.startswith("--config="):
            config_file = arg.split("=", 1)[1]
        elif arg.startswith("--output-dir="):
            output_dir = arg.split("=", 1)[1]
        elif not arg.startswith("--"):
            base_data_dir = arg
    
    # Load configuration
    if config_file:
        config = load_config_with_yaml_support(config_file)
    else:
        config = PipelineConfig()
    
    # Override with command line args
    if output_dir:
        config.base_data_dir = output_dir
    elif base_data_dir:
        config.base_data_dir = base_data_dir
    if generate_frontend_flag:
        config.generate_frontend = True
    if debug_flag:
        config.logging.level = "DEBUG"
    
    # Extract observation ID first (needed for logging setup)
    observation_id = extract_observation_id(ecsv_file)
    
    # Setup comprehensive logging
    logger = setup_pipeline_logging(config, observation_id)
    logger.info(f"=== Transient Pipeline Started ===")
    logger.info(f"Processing files: {Path(ecsv_file).name}, {Path(fits_file).name}")
    logger.info(f"Observation ID: {observation_id}")
    logger.info(f"Log files will be at: {Path(config.base_data_dir) / 'logs'}")
    
    # Setup catalog cache
    setup_catalog_cache(config.caching.cache_dir)
    
    # Validate input files
    if not Path(ecsv_file).exists():
        logger.error(f"ECSV file not found: {ecsv_file}")
        sys.exit(1)
    
    if not Path(fits_file).exists():
        logger.error(f"FITS file not found: {fits_file}")
        sys.exit(1)
    
    # Setup observation directory
    obs_dir = setup_observation_directory(config.base_data_dir, observation_id)
    logger.info(f"Working directory: {obs_dir}")
    
    # Check if this specific file was already processed
    ecsv_basename = Path(ecsv_file).name
    fits_basename = Path(fits_file).name
    
    already_processed = check_if_already_processed(obs_dir, ecsv_basename)
    logger.info(f"File {ecsv_basename} already processed: {already_processed}")
    
    if already_processed:
        logger.info(f"File {ecsv_basename} already processed for observation {observation_id}")
        logger.info("Skipping analysis - results should already exist")
        
        # Check if results exist
        candidates_file = obs_dir / "candidates.tbl"
        if candidates_file.exists():
            logger.info(f"Existing results found at: {candidates_file}")
        else:
            logger.warning("File marked as processed but no results found")
            logger.info("Will reprocess...")
            # Force reprocessing by treating as new file
            already_processed = False
        
        if already_processed:
            if config.generate_frontend:
                # Check if website exists, only generate if missing
                base_public_dir = config.base_public_dir or Path.home() / "public_html"
                website_dir = Path(base_public_dir) / f"obs_{observation_id}"
                if not (website_dir / "index.html").exists():
                    logger.info("Website missing, generating...")
                    generate_frontend(obs_dir, observation_id, config)
                else:
                    logger.info(f"Website already exists at: {website_dir}")
            return
    
    # Copy files to observation directory
    ecsv_dest = obs_dir / ecsv_basename
    fits_dest = obs_dir / fits_basename
    
    # Copy new files if not already present
    if not ecsv_dest.exists():
        shutil.copy(ecsv_file, ecsv_dest)
        logger.info(f"Copied {ecsv_basename} to observation directory")
    
    if not fits_dest.exists():
        shutil.copy(fits_file, fits_dest)
        logger.info(f"Copied {fits_basename} to observation directory")
    
    # Load existing detection tables and metadata
    logger.info("Loading existing detection tables...")
    detection_tables, processed_files = get_existing_detection_tables(obs_dir)
    
    # Process new ECSV file
    new_detection_added = False
    try:
        new_detection = open_ecsv_file(str(ecsv_dest), verbose=True)
        if new_detection is not None and new_detection.meta is not None:
            detection_tables.append(new_detection)
            update_metadata(obs_dir, processed_files, ecsv_basename)
            new_detection_added = True
            logger.info(f"Added new detection table: {ecsv_basename}")
        else:
            logger.error(f"Could not process {ecsv_basename}")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to process {ecsv_basename}: {e}")
        sys.exit(1)
    
    if not detection_tables:
        logger.error("No valid detection tables found")
        sys.exit(1)
    
    logger.info(f"Total detection tables: {len(detection_tables)}")
    
    # Determine if we should run the analysis
    should_analyze, reason = should_run_analysis(obs_dir, new_detection_added)
    logger.info(f"Analysis decision: {reason}")
    
    if should_analyze:
        # Create analysis lock to coordinate with other processes
        analysis_lock = create_analysis_lock(obs_dir)
        
        try:
            first_det = detection_tables[0]
            
            # Run the analysis
            reliable_candidates = process_observation(obs_dir, detection_tables, first_det, config)
            logger.info(f"Analysis completed successfully")
            
            # Generate frontend after successful analysis (if requested)
            if config.generate_frontend:
                logger.info("Generating frontend...")
                generate_frontend(obs_dir, observation_id, config)
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            sys.exit(1)
        finally:
            # Always remove the lock
            remove_analysis_lock(analysis_lock)
    else:
        candidates_file = obs_dir / "candidates.tbl"
        logger.info("No new data to process and results already exist")
        logger.info(f"Existing results: {candidates_file}")
        
        # Only generate frontend if requested
        if config.generate_frontend:
            base_public_dir = config.base_public_dir or Path.home() / "public_html"
            website_dir = Path(base_public_dir) / f"obs_{observation_id}"
            if not (website_dir / "index.html").exists():
                logger.info("Website missing, generating...")
                generate_frontend(obs_dir, observation_id, config)
            else:
                logger.info(f"Website already exists at: {website_dir}")
    
    logger.info(f"=== Pipeline Completed Successfully ===")
    logger.info(f"Observation: {observation_id}")
    logger.info(f"Results directory: {obs_dir}")
    logger.info(f"Log files: {Path(config.base_data_dir) / 'logs' / f'pipeline_{observation_id}.log'}")

if __name__ == "__main__":
    main()
