# Multiplexed Biosensor Analysis Pipeline

A computational pipeline for analyzing and modeling multiplexed bacterial biosensor data.

## Overview

This project provides tools for:
- Processing and analyzing multi-channel biosensor data from different plate readers
- Fitting mechanistic ODE models to sensor responses 
- Training VAE-MLP models for concentration prediction

## Key Components

### Data Processing (`data_loading_v5.py`)
- Handles data import from multiple plate reader formats
- Converts between different reader calibrations
- Processes raw fluorescence and OD measurements
- Supports various biosensor combinations (IPTG-aTc, TTR-THS, etc.)

### Mechanistic Modeling (`mechanistic_model.py`) 
- Implements ODE-based models of biosensor dynamics
- Fits model parameters using least squares optimization
- Accounts for:
  - Cell growth
  - Protein expression
  - Cross-talk between sensors
  - Input-dependent responses

### Deep Learning (`VAE.py`, `VAEMLP.py`)
- Variational autoencoder for dimensionality reduction
- MLP for concentration prediction
- Combined training pipeline

## Usage

1. Data should be organized in Excel files with appropriate metadata
2. Use `define_metadata()` to specify dataset parameters
3. Process raw data through the loading pipeline
4. Fit mechanistic models or train ML models as needed

## File Structure
```
├── data_loading_v5.py    # Data processing pipeline
├── mechanistic_model.py  # ODE model implementation  
├── crosstalk.py         # Sensor specificity analysis
├── VAE.py              # Variational autoencoder
├── VAEMLP.py           # Combined VAE-MLP model
└── notebooks/          # Analysis notebooks
```
Citing
If you use this code in your research, please cite [paper reference].

License
[Add license information]

Authors
[Add author information]