#!/usr/bin/env python3
"""
Clean GRB strategy system that integrates with the exposure calculator.
Uses empirical photometry model for accurate SNR calculations.
"""

import numpy as np
from scipy.optimize import fsolve
from astropy.table import Table

# Parameters from exposure calculator
APE = 6.146      # Fitted transition parameter
GAIN = 0.81      # CCD gain  
ZERO = 10        # Zero offset
RN = 8.0         # Effective readout noise (electrons)

def sbl(A, B, N, x):
    """Smooth transition function between photon and background noise regimes."""
    return A*x + (B-A)*(abs(N)*np.sqrt(1.0 + x*x/(N*N)) + x)/2.0

def break_magnitude(bgsigma, fwhm):
    """Calculate transition magnitude where photon noise equals background noise."""
    return -2.5*np.log10(APE * np.pi/4 * fwhm*fwhm * (bgsigma*GAIN)**2) + ZERO

def log_magerror(magnitude, bgsigma, fwhm):
    """Predict log10(magnitude error) for given magnitude, background, and seeing."""
    break_mag = break_magnitude(bgsigma, fwhm)
    return sbl(0.2, 0.4, 2.5, magnitude - break_mag) + 0.2*break_mag - 2

def magerror_to_snr(magerror):
    """Convert magnitude error to SNR."""
    return 1.0 / (magerror * np.log(10) / 2.5)

def snr_to_magerror(snr):
    """Convert SNR to magnitude error."""
    return 1.0 / (snr * np.log(10) / 2.5)

def bgsigma_from_sky_brightness(sky_brightness, exptime):
    """Calculate bgsigma for given exposure time and sky brightness."""
    return np.sqrt(sky_brightness * exptime + GAIN**2 * RN**2)

def calculate_snr_for_conditions(magnitude, exptime, sky_1s, fwhm, magzero_1s, filter_transmission=1.0):
    """
    Calculate SNR for given observing conditions using the empirical model.
    
    Parameters:
    -----------
    magnitude : float
        Object magnitude
    exptime : float
        Exposure time (seconds)
    sky_1s : float
        Sky brightness (photons/s/pixel) for 1s exposure
    fwhm : float
        Seeing FWHM (pixels)
    magzero_1s : float
        Magnitude zeropoint for 1s exposure
    filter_transmission : float
        Filter transmission factor (1.0 for no filter, <1.0 for filters)
        
    Returns:
    --------
    snr : float
        Signal-to-noise ratio
    """
    # Account for filter transmission
    effective_exptime = exptime * filter_transmission
    effective_magzero_1s = magzero_1s + 2.5*np.log10(filter_transmission)
    
    # Calculate background for this exposure time
    bgsigma = bgsigma_from_sky_brightness(sky_1s, effective_exptime)
    
    # Zeropoint for this exposure time
    magzero = effective_magzero_1s + 2.5*np.log10(effective_exptime)
    
    # Convert target magnitude to magnitude relative to zeropoint
    mag_relative = magnitude - magzero
    
    # Debug output
    if magnitude == 18.5:  # Only debug for one case to avoid spam
        print(f"  Debug SNR calc: mag={magnitude}, exptime={exptime}, transmission={filter_transmission}")
        print(f"  Debug: effective_exptime={effective_exptime}, effective_magzero_1s={effective_magzero_1s:.2f}")
        print(f"  Debug: bgsigma={bgsigma:.2f}, magzero={magzero:.2f}, mag_relative={mag_relative:.2f}")
    
    # Predict magnitude error using empirical model
    predicted_log_magerror = log_magerror(mag_relative, bgsigma, fwhm)
    predicted_magerror = 10**predicted_log_magerror
    
    if magnitude == 18.5:
        print(f"  Debug: predicted_log_magerror={predicted_log_magerror:.3f}, predicted_magerror={predicted_magerror:.4f}")
    
    # Convert to SNR
    snr = magerror_to_snr(predicted_magerror)
    
    if magnitude == 18.5:
        print(f"  Debug: final SNR={snr:.1f}")
    
    return snr

def calculate_required_exptime(magnitude, target_snr, sky_1s, fwhm, magzero_1s, 
                             filter_transmission=1.0, max_exptime=1200):
    """
    Calculate exposure time needed to achieve target SNR.
    
    Parameters:
    -----------
    magnitude : float
        Object magnitude
    target_snr : float
        Desired SNR
    sky_1s : float
        Sky brightness (photons/s/pixel)
    fwhm : float
        Seeing FWHM (pixels)  
    magzero_1s : float
        Magnitude zeropoint for 1s exposure
    filter_transmission : float
        Filter transmission factor
    max_exptime : float
        Maximum allowed exposure time
        
    Returns:
    --------
    exptime : float
        Required exposure time (seconds), or max_exptime if not achievable
    """
    target_magerror = snr_to_magerror(target_snr)
    
    def equation(log_exptime):
        exptime = 10**log_exptime
        if exptime > max_exptime:
            return 1e10  # Large penalty for exceeding max time
            
        effective_exptime = exptime * filter_transmission
        effective_magzero_1s = magzero_1s + 2.5*np.log10(filter_transmission)
        
        bgsigma = bgsigma_from_sky_brightness(sky_1s, effective_exptime)
        magzero = effective_magzero_1s + 2.5*np.log10(effective_exptime)
        mag_relative = magnitude - magzero
        
        predicted_log_magerror = log_magerror(mag_relative, bgsigma, fwhm)
        predicted_magerror = 10**predicted_log_magerror
        
        return predicted_magerror - target_magerror
    
    try:
        # Try to solve for required exposure time
        log_exptime_solution = fsolve(equation, np.log10(100))[0]  # Start at 100s
        exptime = 10**log_exptime_solution
        
        # Check if solution is reasonable
        if exptime > max_exptime or exptime < 1:
            return max_exptime
        return exptime
    except:
        return max_exptime

def load_observing_conditions(ecsv_file='image.ecsv'):
    """
    Load observing conditions from ECSV file metadata.
    
    Parameters:
    -----------
    ecsv_file : str
        Path to the ECSV file containing observing metadata
        
    Returns:
    --------
    dict : Dictionary containing sky_1s, fwhm, magzero_1s, and exposure time
    """
    try:
        # Read the ECSV file
        table = Table.read(ecsv_file)
        
        # Extract parameters from metadata
        fwhm = table.meta['FWHM']  # FWHM in pixels
        bgsigma = table.meta['BGSIGMA']  # Background sigma in counts
        magzero_raw = table.meta['MAGZERO']  # Raw magnitude zeropoint (reduced by 10)
        exposure = table.meta['EXPOSURE']  # Exposure time in seconds
        
        # Convert to the format needed by our strategy functions
        # BGSIGMA is the background noise in ADU for this exposure
        # We need to convert this to sky brightness in ADU/s/pixel
        
        # The background noise comes from: bgsigma^2 = sky_rate * exptime + (readnoise/gain)^2
        # where sky_rate is in ADU/s/pixel and readnoise is in electrons
        readnoise_adu = RN / GAIN  # Convert readnoise to ADU
        
        # Solve for sky_rate: sky_rate = (bgsigma^2 - readnoise_adu^2) / exptime
        sky_1s = (bgsigma**2 - readnoise_adu**2) / exposure
        
        # Ensure sky_1s is positive
        sky_1s = max(sky_1s, 1.0)
        
        # Correct the zeropoint - the +10 correction might be wrong
        # Let's use the raw value and see if it makes more sense
        magzero = magzero_raw  # Try using the raw zeropoint first
        
        # Convert magzero to 1-second zeropoint
        magzero_1s = magzero - 2.5*np.log10(exposure)
        
        conditions = {
            'sky_1s': sky_1s,
            'fwhm': fwhm, 
            'magzero_1s': magzero_1s,
            'exposure': exposure,
            'bgsigma': bgsigma,
            'magzero': magzero,
            'magzero_raw': magzero_raw,
            'readnoise_adu': readnoise_adu
        }
        
        print(f"Debug: bgsigma={bgsigma:.2f} ADU, readnoise_adu={readnoise_adu:.2f} ADU")
        print(f"Debug: calculated sky_1s={sky_1s:.2f} ADU/s/pixel")
        
        return conditions
        
    except Exception as e:
        print(f"Warning: Could not load conditions from {ecsv_file}: {e}")
        print("Using default values...")
        return {
            'sky_1s': 11.5,
            'fwhm': 2.1,
            'magzero_1s': 11.28,  # Back to original value
            'exposure': 120.0,
            'bgsigma': 46.6,
            'magzero': 16.48,
            'magzero_raw': 16.48,
            'readnoise_adu': RN / GAIN
        }

def determine_grb_strategy(magnitude, time_since_trigger, ecsv_file='image.ecsv', 
                          sky_1s=None, fwhm=None, magzero_1s=None):
    """
    Determine optimal GRB observing strategy using proper SNR calculations.
    
    Parameters:
    -----------
    magnitude : float
        Estimated GRB magnitude
    time_since_trigger : float
        Time since GRB trigger (seconds)
    ecsv_file : str
        Path to ECSV file containing observing conditions (optional)
    sky_1s : float
        Sky brightness override (photons/s/pixel) - if None, loads from ECSV
    fwhm : float
        Seeing FWHM override (pixels) - if None, loads from ECSV
    magzero_1s : float
        Magnitude zeropoint override for 1s exposure - if None, loads from ECSV
        
    Returns:
    --------
    dict : Strategy recommendation with all details
    """
    
    # Load observing conditions from ECSV file if not provided
    if sky_1s is None or fwhm is None or magzero_1s is None:
        conditions = load_observing_conditions(ecsv_file)
        if sky_1s is None:
            sky_1s = conditions['sky_1s']
        if fwhm is None:
            fwhm = conditions['fwhm']
        if magzero_1s is None:
            magzero_1s = conditions['magzero_1s']
    
    # Define observing configurations
    configs = {
        'emccd_no_filter': {'use_emccd': True, 'num_filters': 0, 'transmission': 1.0, 'min_snr': 15},
        'emccd_with_filter': {'use_emccd': True, 'num_filters': 1, 'transmission': 0.5, 'min_snr': 15},
        'multifilter_RGIZ': {'use_emccd': False, 'num_filters': 4, 'transmission': 0.2875, 'min_snr': 10},
        'single_filter': {'use_emccd': False, 'num_filters': 1, 'transmission': 0.5, 'min_snr': 5},
        'clear_only': {'use_emccd': False, 'num_filters': 0, 'transmission': 1.0, 'min_snr': 5},
    }
    
    # Determine maximum exposure time based on time since trigger
    if time_since_trigger < 300:  # < 5 minutes
        max_exptime = 30
    elif time_since_trigger < 600:  # < 10 minutes  
        max_exptime = 60
    elif time_since_trigger < 1800:  # < 30 minutes
        max_exptime = 200
    else:
        max_exptime = 1200  # 20 minutes max
    
    best_config = None
    best_utility = -1
    
    for config_name, config in configs.items():
        # EMCCD has special constraints
        if config['use_emccd']:
            # EMCCD saturates at ~42 photons/frame (1/30s exposure)
            max_emccd_exptime = min(30, max_exptime)
            
            # Check if we can achieve minimum SNR with EMCCD
            snr = calculate_snr_for_conditions(
                magnitude, max_emccd_exptime, sky_1s, fwhm, magzero_1s, 
                config['transmission']
            )
            
            if snr < config['min_snr']:
                continue  # Skip this configuration
                
            exptime = max_emccd_exptime
            
        else:
            # Regular CCD - calculate required exposure time
            exptime = calculate_required_exptime(
                magnitude, config['min_snr'], sky_1s, fwhm, magzero_1s,
                config['transmission'], max_exptime
            )
            
            # Verify we can achieve the minimum SNR
            snr = calculate_snr_for_conditions(
                magnitude, exptime, sky_1s, fwhm, magzero_1s,
                config['transmission']
            )
            
            if snr < config['min_snr'] * 0.9:  # Allow 10% tolerance
                continue
        
        # Calculate utility (prioritize multi-filter observations)
        if config['num_filters'] > 1:
            utility = snr * config['num_filters'] * 2  # Bonus for multi-filter
        else:
            utility = snr
            
        if utility > best_utility:
            best_utility = utility
            best_config = {
                'config_name': config_name,
                'exp_time': exptime,
                'use_emccd': config['use_emccd'],
                'num_filters': config['num_filters'],
                'transmission': config['transmission'],
                'snr': snr,
                'utility': utility,
                'min_snr_required': config['min_snr']
            }
    
    if best_config is None:
        # Fallback - longest exposure with clear filter
        exptime = max_exptime
        snr = calculate_snr_for_conditions(magnitude, exptime, sky_1s, fwhm, magzero_1s, 1.0)
        best_config = {
            'config_name': 'fallback_clear',
            'exp_time': exptime,
            'use_emccd': False,
            'num_filters': 0,
            'transmission': 1.0,
            'snr': snr,
            'utility': snr,
            'min_snr_required': 1.0
        }
    
    # Add metadata
    best_config.update({
        'magnitude': magnitude,
        'time_since_trigger': time_since_trigger,
        'max_allowed_exp_time': max_exptime,
        'background_conditions': f'sky_1s={sky_1s} ph/s/px, FWHM={fwhm}px',
        'magzero_1s': magzero_1s
    })
    
    return best_config

def test_grb_scenarios():
    """Test the strategy determination with realistic scenarios."""
    
    print("=== GRB Strategy Test ===\n")
    
    # Load current observing conditions
    conditions = load_observing_conditions()
    print(f"Loaded conditions: FWHM={conditions['fwhm']:.2f}px, " + 
          f"sky_1s={conditions['sky_1s']:.1f} ph/s/px, " +
          f"magzero_1s={conditions['magzero_1s']:.2f}")
    print(f"Reference exposure: {conditions['exposure']:.0f}s, " +
          f"bgsigma={conditions['bgsigma']:.1f}, magzero={conditions['magzero']:.2f} " +
          f"(raw: {conditions['magzero_raw']:.2f})\n")
    
    # Test scenarios
    scenarios = [
        {'mag': 16.0, 'time': 600, 'desc': 'Bright object, 16.0 mag, 10min after trigger'},
        {'mag': 18.5, 'time': 1200, 'desc': 'Moderate object, 18.5 mag 20min after trigger'}, 
        {'mag': 20.0, 'time': 3600, 'desc': 'Faint object, 20.0 mag, 1hr after trigger'},
        {'mag': 22.0, 'time': 7200, 'desc': 'Very faint object, 22.0 mag, 30min after trigger'},
    ]
    
    for scenario in scenarios:
        print(f"{scenario['desc']}")
        result = determine_grb_strategy(scenario['mag'], scenario['time'])
        
        print(f"  Strategy: {result['config_name']}")
        print(f"  Exposure: {result['exp_time']:.1f}s, EMCCD: {result['use_emccd']}")
        print(f"  Filters: {result['num_filters']}, SNR: {result['snr']:.1f}")
        print(f"  Utility: {result['utility']:.1f}")
        print(f"  Transmission: {result['transmission']:.3f}")
        print()

if __name__ == "__main__":
    # Run tests
    test_grb_scenarios()
    
    print("=== Usage Example ===")
    print("result = determine_grb_strategy(magnitude=18.5, time_since_trigger=600)")
    print("print(f'Strategy: {result[\"config_name\"]}, SNR: {result[\"snr\"]:.1f}')")
    
    # Example usage
    result = determine_grb_strategy(magnitude=18.5, time_since_trigger=600)
    print(f"\nExample result: Strategy: {result['config_name']}, SNR: {result['snr']:.1f}")
