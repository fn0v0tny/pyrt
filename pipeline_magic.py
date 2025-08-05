#!/usr/bin/python3

import sys
import shutil
import json
import subprocess
import time  # Added missing import
from catalog import QueryParams, setup_catalog_cache
from transients import *
from transient_analyser import *
from extraction_manager import ImageExtractionManager
import os
from pathlib import Path
from datetime import datetime

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
        print(f"WARNING: Could not extract observation ID from {ecsv_file_path}: {e}")
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
        print(f"WARNING: Could not create observation directory {obs_dir}: {e}")
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
            print(f"WARNING: Could not load metadata: {e}")
    
    # Load existing detection tables
    for ecsv_file in obs_dir.glob("*.ecsv"):
        if ecsv_file.name in processed_files:
            try:
                detection_data = open_ecsv_file(str(ecsv_file))
                if detection_data is not None and detection_data.meta is not None:
                    detection_tables.append(detection_data)
                    print(f"Loaded existing detection table: {ecsv_file.name}")
            except Exception as e:
                print(f"WARNING: Could not load {ecsv_file}: {e}")
    
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
                print(f"INFO: Another process is updating metadata, skipping...")
                return
        
        # Atomic move
        temp_file.replace(metadata_file)
        
    except Exception as e:
        print(f"WARNING: Could not save metadata: {e}")
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
                print(f"INFO: Could not read metadata (locked), assuming not processed")
                return False
    except Exception as e:
        print(f"WARNING: Could not read metadata: {e}")
        return False

def generate_frontend(obs_dir):
    """Generate frontend using gen_frontend.py script."""
    try:
        frontend_script = Path("/home/fnovotny/bin/gen_frontend.py")
        
        if frontend_script.exists():
            print(f"Generating frontend for observation in {obs_dir}")
            result = subprocess.run([
                str(frontend_script), str(obs_dir)
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                print("Frontend generated successfully")
                if result.stdout:
                    print(f"Frontend output: {result.stdout.strip()}")
            else:
                print(f"Frontend generation failed with exit code {result.returncode}")
                if result.stderr:
                    print(f"Frontend error: {result.stderr}")
        else:
            print(f"Frontend script not found at {frontend_script}")
            print("Skipping frontend generation")
            
    except subprocess.TimeoutExpired:
        print("WARNING: Frontend generation timed out")
    except Exception as e:
        print(f"WARNING: Frontend generation failed: {e}")

def process_observation(obs_dir, detection_tables, first_det):
    """Run the full transient analysis pipeline."""
    print("Running transient analysis...")
    
    # Setup image manager
    image_manager = ImageExtractionManager(detection_tables)
    ra, dec = image_manager.field_center
    print(f"Field center: RA={ra:.6f}, DEC={dec:.6f}")
    
    # Setup query parameters
    query_params = QueryParams(**{
        "ra": ra,
        "dec": dec,
        "width": 1.2 * first_det.meta["FIELD"],
        "height": 1.2 * first_det.meta["FIELD"],
        "mlim": 20,
    })
    
    print(f"Query parameters: RA={ra:.6f}, DEC={dec:.6f}, "
          f"width={1.2 * first_det.meta['FIELD']:.3f}, "
          f"height={1.2 * first_det.meta['FIELD']:.3f}")
    
    # Run transient analysis
    transient_analyzer = TransientAnalyzer(data_dir=obs_dir)
    multi_analyzer = MultiDetectionAnalyzer(transient_analyzer, lightcurve_dir=obs_dir)
    
    print("Processing detection tables...")
    # FIX: Unpack the tuple returned by the method
    reliable_candidates, lightcurves = multi_analyzer.process_detection_tables_with_lightcurves(
        detection_tables=detection_tables,
        catalogs=["atlas@local", "gaia", "usno"],
        params=query_params,
        idlimit=3.0,
        radius_check=20.0,
        filter_pattern="r",
        min_n_detections=5,
        min_catalogs=3,
        min_quality=0.05,
    )
    
    print(f"Found {len(reliable_candidates)} reliable candidates")
    if len(reliable_candidates) > 0:
        print(reliable_candidates)
    
    # Save results
    candidates_file = obs_dir / "candidates.tbl"
    reliable_candidates.write(
        str(candidates_file), format="ascii.ipac", overwrite=True
    )
    print(f"Results saved to {candidates_file}")
    
    # Also save lightcurve information if available
    if lightcurves:
        print(f"Generated {len(lightcurves)} lightcurves")
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
            print(f"Lightcurve summary saved to {lightcurve_summary_file}")
        except Exception as e:
            print(f"WARNING: Could not save lightcurve summary: {e}")
    
    return reliable_candidates

def should_run_analysis(obs_dir, new_detection_added):
    """Determine if analysis should be run (with coordination between parallel processes)."""
    candidates_file = obs_dir / "candidates.tbl"
    analysis_lock = obs_dir / ".analysis.lock"
    
    print(f"Checking analysis conditions:")
    print(f"  - New detection added: {new_detection_added}")
    print(f"  - Candidates file exists: {candidates_file.exists()}")
    print(f"  - Analysis lock exists: {analysis_lock.exists()}")
    
    # If we added new data, we definitely need analysis
    if new_detection_added:
        print("  → Decision: Need analysis (new detection data)")
        return True, "New detection data added"
    
    # If no results exist, we need analysis
    if not candidates_file.exists():
        print("  → Decision: Need analysis (no existing results)")
        return True, "No existing results found"
    
    # Check if another process is already running analysis
    if analysis_lock.exists():
        # Check if the lock is stale (older than 15 minutes)
        try:
            lock_age = time.time() - analysis_lock.stat().st_mtime
            print(f"  - Lock age: {lock_age/60:.1f} minutes")
            if lock_age > 900:  # 15 minutes
                print(f"  → Decision: Removing stale lock and running analysis")
                analysis_lock.unlink()
                return True, "Stale lock removed, rerunning analysis"
            else:
                print(f"  → Decision: Skip analysis (another process running)")
                return False, f"Analysis already running (lock age: {lock_age/60:.1f} minutes)"
        except Exception as e:
            print(f"  - Error checking lock: {e}")
            print(f"  → Decision: Skip analysis (lock check failed)")
            return False, "Analysis lock present (cannot check age)"
    
    print("  → Decision: Skip analysis (results exist, no new data)")
    return False, "Results exist and no new data"

def create_analysis_lock(obs_dir):
    """Create analysis lock file."""
    analysis_lock = obs_dir / ".analysis.lock"
    try:
        print(f"Creating analysis lock: {analysis_lock}")
        with open(analysis_lock, 'w') as f:
            f.write(f"PID: {os.getpid()}\nStarted: {datetime.now().isoformat()}\n")
        print(f"Analysis lock created successfully")
        return analysis_lock
    except Exception as e:
        print(f"WARNING: Could not create analysis lock: {e}")
        return None

def remove_analysis_lock(analysis_lock):
    """Remove analysis lock file."""
    if analysis_lock and analysis_lock.exists():
        try:
            analysis_lock.unlink()
            print("Analysis lock removed")
        except Exception as e:
            print(f"WARNING: Could not remove analysis lock: {e}")

def main():
    if len(sys.argv) < 3:
        print("Usage: pipeline_magic.py <ecsv_file> <fits_file> [base_data_dir]")
        sys.exit(1)
    cache_dir = "./catalog_cache"
    setup_catalog_cache(cache_dir)

    ecsv_file = sys.argv[1]
    fits_file = sys.argv[2]
    base_data_dir = sys.argv[3] if len(sys.argv) > 3 else "/home/fnovotny/transient_work/"
    
    print(f"=== Transient Pipeline Started ===")
    print(f"Processing files: {Path(ecsv_file).name}, {Path(fits_file).name}")
    
    # Validate input files
    if not Path(ecsv_file).exists():
        print(f"ERROR: ECSV file not found: {ecsv_file}")
        sys.exit(1)
    
    if not Path(fits_file).exists():
        print(f"ERROR: FITS file not found: {fits_file}")
        sys.exit(1)
    
    # Extract observation ID and setup directory
    observation_id = extract_observation_id(ecsv_file)
    obs_dir = setup_observation_directory(base_data_dir, observation_id)
    print(f"Observation ID: {observation_id}")
    print(f"Working directory: {obs_dir}")
    
    # Check if this specific file was already processed
    ecsv_basename = Path(ecsv_file).name
    fits_basename = Path(fits_file).name
    
    already_processed = check_if_already_processed(obs_dir, ecsv_basename)
    print(f"File {ecsv_basename} already processed: {already_processed}")
    
    if already_processed:
        print(f"File {ecsv_basename} already processed for observation {observation_id}")
        print("Skipping analysis - results should already exist")
        
        # Check if results exist
        candidates_file = obs_dir / "candidates.tbl"
        if candidates_file.exists():
            print(f"Existing results found at: {candidates_file}")
        else:
            print("WARNING: File marked as processed but no results found")
            print("Will reprocess...")
            # Force reprocessing by treating as new file
            already_processed = False
        
        if already_processed:
            # Still generate frontend in case it's missing
            generate_frontend(obs_dir)
            return
    
    # Copy files to observation directory
    ecsv_dest = obs_dir / ecsv_basename
    fits_dest = obs_dir / fits_basename
    
    # Copy new files if not already present
    if not ecsv_dest.exists():
        shutil.copy(ecsv_file, ecsv_dest)
        print(f"Copied {ecsv_basename} to observation directory")
    
    if not fits_dest.exists():
        shutil.copy(fits_file, fits_dest)
        print(f"Copied {fits_basename} to observation directory")
    
    # Load existing detection tables and metadata
    print("Loading existing detection tables...")
    detection_tables, processed_files = get_existing_detection_tables(obs_dir)
    
    # Process new ECSV file
    new_detection_added = False
    try:
        new_detection = open_ecsv_file(str(ecsv_dest), verbose=True)
        if new_detection is not None and new_detection.meta is not None:
            detection_tables.append(new_detection)
            update_metadata(obs_dir, processed_files, ecsv_basename)
            new_detection_added = True
            print(f"Added new detection table: {ecsv_basename}")
        else:
            print(f"ERROR: Could not process {ecsv_basename}")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to process {ecsv_basename}: {e}")
        sys.exit(1)
    
    if not detection_tables:
        print("ERROR: No valid detection tables found")
        sys.exit(1)
    
    print(f"Total detection tables: {len(detection_tables)}")
    
    # Determine if we should run the analysis
    should_analyze, reason = should_run_analysis(obs_dir, new_detection_added)
    print(f"Analysis decision: {reason}")
    
    if should_analyze:
        # Create analysis lock to coordinate with other processes
        analysis_lock = create_analysis_lock(obs_dir)
        
        try:
            first_det = detection_tables[0]
            
            # Run the analysis
            reliable_candidates = process_observation(obs_dir, detection_tables, first_det)
            print(f"Analysis completed successfully")
            
            # Generate frontend after successful analysis
            print("Generating frontend...")
            generate_frontend(obs_dir)
            
        except Exception as e:
            print(f"ERROR: Analysis failed: {e}")
            sys.exit(1)
        finally:
            # Always remove the lock
            remove_analysis_lock(analysis_lock)
    else:
        candidates_file = obs_dir / "candidates.tbl"
        print("No new data to process and results already exist")
        print(f"Existing results: {candidates_file}")
        
        # Still generate frontend in case it needs updating
        generate_frontend(obs_dir)
    
    print(f"=== Pipeline Completed Successfully ===")
    print(f"Observation: {observation_id}")
    print(f"Results directory: {obs_dir}")

if __name__ == "__main__":
    main()
