# pyrt: Astronomical Transient Detection Pipeline

A comprehensive Python suite for photometric calibration and transient detection in astronomical FITS images.

## Overview

pyrt provides tools for:
- Photometric calibration of astronomical images
- Transient object detection and analysis
- Multi-catalog cross-matching and verification 
- Time-series analysis of variable sources
- Hot pixel and artifact filtering

## Features

### Catalog Support
- ATLAS catalog (local and VizieR)
- Gaia DR3
- Pan-STARRS DR2 
- USNO-B1.0
- Custom local catalogs

### Detection Capabilities
- Multi-epoch transient detection
- Moving object identification
- Variable source analysis
- Quality scoring and filtering
- Contextual analysis using multiple reference catalogs

### Analysis Tools
- Shape parameter analysis
- Photometric feature extraction
- Source density estimation
- Quality metrics computation
- Cross-matching across multiple epochs

## Installation

### Prerequisites
- Python 3.8 or higher
- astropy
- numpy
- scipy
- scikit-learn
- Required Python packages listed in `requirements.txt`

### Basic Installation
```bash
git clone https://github.com/yourusername/pyrt.git
cd pyrt
pip install -r requirements.txt
```

## Usage

### Basic Transient Detection
```python
from catalog import QueryParams
from transient_analyser import TransientAnalyzer

# Initialize analyzer
analyzer = TransientAnalyzer()

# Set up query parameters
params = QueryParams(
    ra=180.0,  # Center RA in degrees
    dec=30.0,  # Center Dec in degrees
    width=0.5, # Field width in degrees
    height=0.5,# Field height in degrees
    mlim=20.0  # Magnitude limit
)

# Process detections
results = analyzer.find_transients_multicatalog(
    detections=your_detection_table,
    catalogs=['atlas@local', 'gaia', 'usno'],
    params=params,
    idlimit=2.0
)
```

### Multi-Epoch Analysis
```python
from transient_analyser import MultiDetectionAnalyzer

# Initialize multi-epoch analyzer
multi_analyzer = MultiDetectionAnalyzer(TransientAnalyzer())

# Process multiple epochs
candidates = multi_analyzer.process_detection_tables(
    detection_tables=detection_tables,
    catalogs=['atlas@local', 'gaia', 'usno'],
    min_n_detections=3,
    min_catalogs=2
)
```

## Code Structure

- `transients.py`: Core transient detection functionality
- `catalog.py`: Catalog access and management
- `transient_analyser.py`: Advanced analysis tools
- `process_transients.py`: Pipeline execution script