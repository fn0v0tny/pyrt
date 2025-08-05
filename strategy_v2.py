#!/usr/bin/env python3
"""
Clean GRB strategy system that integrates with the exposure calculator.
Uses empirical photometry model for accurate SNR calculations.
"""

import numpy as np
from scipy.optimize import fsolve

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
    
    # Predict magnitude error using empirical model
    predicted_log_magerror = log_magerror(mag_relative, bgsigma, fwhm)
    predicted_magerror = 10**predicted_log_magerror
    
    # Convert to SNR
    snr = magerror_to_snr(predicted_magerror)
    
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

def determine_grb_strategy(magnitude, time_since_trigger, sky_1s=11.5, fwhm=2.1, magzero_1s=11.28):
    """
    Determine optimal GRB observing strategy using proper SNR calculations.
    
    Parameters:
    -----------
    magnitude : float
        Estimated GRB magnitude
    time_since_trigger : float
        Time since GRB trigger (seconds)
    sky_1s : float
        Sky brightness (photons/s/pixel)
    fwhm : float
        Seeing FWHM (pixels)
    magzero_1s : float
        Magnitude zeropoint for 1s exposure
        
    Returns:
    --------
    dict : Strategy recommendation with all details
    """
    
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
    
    # Test scenarios
    scenarios = [
        {'mag': 16.0, 'time': 600, 'desc': 'Bright object, 10min after trigger'},
        {'mag': 18.5, 'time': 300, 'desc': 'Moderate object, 5min after trigger'}, 
        {'mag': 20.0, 'time': 3600, 'desc': 'Faint object, 1hr after trigger'},
        {'mag': 22.0, 'time': 1800, 'desc': 'Very faint object, 30min after trigger'},
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
