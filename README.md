# MVPA Searchlight Analysis for Music vs Non-Music Decoding

A Python implementation of multivariate pattern analysis (MVPA) using searchlight approach to decode music vs non-music stimuli from fMRI data. This analysis uses the Nilearn library and includes robust statistical comparisons against both dummy classifiers and theoretical chance levels.

## Overview

This script performs a searchlight analysis to identify brain regions that can reliably distinguish between musical and non-musical emotional stimuli using fMRI data. The analysis includes:

- Support Vector Machine (SVM) classification with searchlight
- Comparison against dummy classifier baseline
- Statistical significance testing with multiple correction methods
- Comprehensive visualization of results

## Dataset

This analysis uses beta images derived from the openly available dataset:

**Lepping, R. J., et al. (2016). Neural Processing of Emotional Musical and Nonmusical Stimuli in Depression. PLoS ONE, 11(6), e0156859.** https://doi.org/10.1371/journal.pone.0156859

The dataset is hosted on OpenNeuro and can be accessed here: https://openneuro.org/datasets/ds000171

## Requirements

```python
numpy
pandas
nilearn
scikit-learn
matplotlib
scipy
joblib
```



## Usage

### Basic Usage

```python
python searchlight_analysis.py
```

### Configuration

Update the paths in the `main()` function:

```python
csv_path = "path/to/your/searchlight_data.csv"
mask_path = "path/to/your/mask.nii"
output_dir = "path/to/output/directory"
```

### Input Data Format

The script expects a CSV file with the following columns:
- `beta_file`: Path to beta map NIfTI files
- `group`: Subject group (script filters for 'control')
- `target`: Stimulus type (1,3 = music; 2,4 = non-music)
- `subject`: Subject ID for cross-validation

## Analysis Pipeline

### 1. Data Preparation
- Loads beta maps and filters for control group
- Recodes targets: 0 = Music, 1 = Non-music
- Applies brain mask to constrain analysis

### 2. Searchlight Analysis
- **Radius**: 4mm spherical searchlight
- **Classifier**: Linear SVM (LinearSVC)
- **Baseline**: Stratified dummy classifier
- **Cross-validation**: Leave-one-subject-out
- **Metric**: Area Under ROC Curve (AUC)

### 3. Statistical Testing
- Comparison against dummy classifier (accounts for class imbalance)
- Comparison against theoretical chance (AUC = 0.5)
- One-tailed tests (SVM > baseline)
- Multiple comparison correction:
  - FDR (False Discovery Rate) correction
  - Cluster-based thresholding (min cluster size = 20 voxels)

### 4. Output Files

The analysis generates multiple output files:

#### NIfTI Maps
- `control_music_svm_auc.nii.gz`: SVM AUC scores for each voxel
- `control_music_dummy_auc.nii.gz`: Dummy classifier AUC scores
- `control_music_diff_from_dummy.nii.gz`: Difference between SVM and dummy
- `control_music_diff_from_chance.nii.gz`: Difference from chance (0.5)
- `control_music_z_vs_dummy.nii.gz`: Z-scores for SVM vs dummy
- `control_music_z_vs_chance.nii.gz`: Z-scores for SVM vs chance
- `control_music_cluster_vs_dummy.nii.gz`: Significant clusters (vs dummy)
- `control_music_cluster_vs_chance.nii.gz`: Significant clusters (vs chance)
- `control_music_threshold_[0.55-0.7].nii.gz`: Thresholded maps at different AUC levels

#### Visualizations
- `control_auc_comparison.png`: Multi-slice comparison of SVM vs dummy AUC
- `control_significance_maps.png`: Statistical significance maps
- `control_summary_statistics.png`: Histograms and summary statistics





## Interpretation

- **AUC > 0.6**: Regions showing above-chance decoding performance
- **AUC > 0.7**: Regions with strong decoding ability
- **Significant clusters**: Regions surviving multiple comparison correction
- **SVM vs Dummy**: Accounts for class imbalance in the data: For this data set we have 3 runs for music and 2 runs for non-msuic 3:2 

## Example Results

The analysis will output:
- Brain regions capable of decoding music vs non-music
- Statistical maps showing significance levels
- Summary statistics including mean AUC and number of significant voxels

## Troubleshooting

### Memory Issues
- Reduce `n_jobs` parameter in SearchLight if running out of memory
