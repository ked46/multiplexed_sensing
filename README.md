# Multiplexed Biosensor Analysis Pipeline

This repository contains all relevant code and data for the paper:
*Engineering microbial consortia for distributed signal processing* by Katherine E. Duncker, Ashwini R. Shende, Irida Shyti, Ashley Ruan, Ryan D'Cunha, Helena R. Ma, Harshitha Venugopal Lavanya, Sizhe Liu, Neil Gottel, Deverick J. Anderson, Claudia K. Gunsch, and Lingchong You

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
- Implements ODE-based models of biosensor dynamicsSynthetic gene circuits are a way to program living cells, much like we program electronic circuits.
The goal of synthetic gene circuits is to create cells that can sense their environment, process information, and respond in a predictable, programmable way. This enables a wide range of applications, from smart therapeutics that only release drugs under specific conditions, to biosensors that detect environmental toxins, to engineered microbes for biomanufacturing. Essentially, we’re treating cells as programmable systems, inspired by electronics but built from the language of biology
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

## Repository Organization

### Main Pipeline Code
- `2025-04-21 full pipeline for any dataset.ipynb`: Main generalizable pipeline that can run any dataset
- Dataset-specific notebooks: `figure_{2,3,4,5,6}...ipynb`
- `evaluate VAE-MLP pipeline.ipynb`: For evaluating trained models

### Data Structure
- **Data Files/**: Raw experimental data (.xlsx, .csv)
  - Time course data with OD and fluorescence measurements
  - Plate reader conversion files for cross-calibration
- **Helenas data analysis/**: Antibiotic-treated experiments data
  - `allexps.zip`: Original data from Ma et al.
  - `process Helena Ma data allexps.ipynb`: Data extraction pipeline
  - `unique_rows.xlsx`: Condition metadata
- **ODE_predictions/aTc_IPTG/**: Model predictions for supplementary figure 14
- **figures/**: Generated figures and plots
- **final_trained_models/**: Saved model parameters
- **parameter_files/**: Model parameters for each dataset
- **sink water experiment analysis/**: Hospital sink water sensing data (Figure 6)

### Using the Pipeline
1. For paper datasets:
   - Use corresponding figure notebooks directly
   - Results match paper figures (except for randomized simulations)
2. For new datasets:
   - Format data following existing structure
   - Modify metadata in `data_processing.py`
   - Adjust pipeline parameters as needed

### Note on VAE-MLP Models
- Stochastic vs deterministic evaluation options available
- For exact paper results, use pre-trained models
- MLP evaluations are deterministic when using saved latent variables

## Dependencies and Installation

### Python Version
- Python 3.8 or higher recommended

### Required Packages
```bash
# Core data processing and scientific computing
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0

# Deep learning
torch>=1.9.0
torchvision>=0.10.0

# Data visualization
matplotlib>=3.4.0
seaborn>=0.11.0

# File handling
openpyxl>=3.0.0  # for Excel file support
xlrd>=2.0.0      # for legacy Excel file support
```