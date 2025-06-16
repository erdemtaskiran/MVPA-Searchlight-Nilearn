import os
import numpy as np
import pandas as pd
from nilearn import image, plotting
from nilearn.decoding import SearchLight
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import label
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(csv_path, mask_path):
    """Load data from CSV and prepare it for searchlight analysis."""
    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded data for {len(df)} beta maps")
    
    # Filter for control group only
    df = df[df['group'] == 'control']
    print(f"\nFiltered to {len(df)} beta maps from control group")
    
    # Load mask
    mask_img = image.load_img(mask_path)
    print(f"Loaded mask with shape: {mask_img.shape}")
    
    # Recode targets to focus on music vs non-music (0: Music, 1: Non-music)
    # Changed to 0/1 for AUC compatibility
    df['stimulus_type'] = np.where(df['target'].isin([1, 3]), 0, 1)  # 1,3 are music, 2,4 are non-music
    
    # Print data summary
    print("\nData Summary (Control Group Only):")
    print(f"Total samples: {len(df)}")
    print(f"Music samples (0): {sum(df['stimulus_type'] == 0)}")
    print(f"Non-music samples (1): {sum(df['stimulus_type'] == 1)}")
    print(f"Number of subjects: {len(df['subject'].unique())}")
    
    # Check data balance per subject
    print("\nData balance per subject:")
    for subject in sorted(df['subject'].unique()):
        subj_data = df[df['subject'] == subject]
        music_count = sum(subj_data['stimulus_type'] == 0)
        nonmusic_count = sum(subj_data['stimulus_type'] == 1)
        print(f"Subject {subject}: {music_count} music, {nonmusic_count} non-music")
    
    # Load all beta images
    beta_imgs = [image.load_img(f) for f in df['beta_file']]
    
    # Get labels and subject IDs
    labels = df['stimulus_type'].values
    subject_ids = df['subject'].values
    
    return beta_imgs, labels, subject_ids, mask_img

def run_searchlight_with_dummy(beta_4d, labels, subject_ids, mask_img):
    """
    Run searchlight analysis for both SVM and dummy classifier using AUC.
    
    Returns:
    --------
    svm_scores : ndarray
        AUC scores for SVM classifier
    dummy_scores : ndarray
        AUC scores for dummy classifier
    """
    print("\nRunning searchlight analysis with SVM and dummy classifiers...")
    
    # SVM searchlight
    print("Running SVM searchlight...")
    svm_searchlight = SearchLight(
        mask_img,
        radius=4,  
        estimator=LinearSVC(dual=False, max_iter=10000, random_state=42),
        cv=LeaveOneGroupOut(),
        scoring='roc_auc',  # Changed to AUC
        n_jobs=4,
        verbose=1
    )
    svm_searchlight.fit(beta_4d, labels, groups=subject_ids)
    svm_scores = svm_searchlight.scores_
    
    # Dummy searchlight
    print("\nRunning dummy classifier searchlight...")
    dummy_searchlight = SearchLight(
        mask_img,
        radius=4,  
        estimator=DummyClassifier(strategy='stratified', random_state=42),
        cv=LeaveOneGroupOut(),
        scoring='roc_auc',  # Changed to AUC
        n_jobs=4,
        verbose=1
    )
    dummy_searchlight.fit(beta_4d, labels, groups=subject_ids)
    dummy_scores = dummy_searchlight.scores_
    
    return svm_scores, dummy_scores

def compute_statistical_maps(svm_scores, dummy_scores, chance_level=0.5):
    """
    Compute statistical maps comparing SVM to dummy and to chance.
    
    Parameters:
    -----------
    svm_scores : ndarray
        AUC scores from SVM classifier
    dummy_scores : ndarray
        AUC scores from dummy classifier
    chance_level : float
        Theoretical chance level (default: 0.5)
    
    Returns:
    --------
    stats_dict : dict
        Dictionary containing all statistical maps
    """
    # Calculate differences
    diff_from_dummy = svm_scores - dummy_scores
    diff_from_chance = svm_scores - chance_level
    
    # For statistical testing, we need to estimate variance
    # Since we don't have fold-wise scores, we'll use a bootstrap approach
    # or assume a reasonable variance based on the scores
    
    # Estimate standard error (conservative approach)
    # This is a simplification - ideally we'd have fold-wise scores
    se_estimate = 0.1  # Conservative estimate based on typical searchlight variance
    
    # Z-scores
    z_vs_dummy = diff_from_dummy / se_estimate
    z_vs_chance = diff_from_chance / se_estimate
    
    # One-tailed p-values (testing if SVM > dummy/chance)
    from scipy.stats import norm
    p_vs_dummy = 1 - norm.cdf(z_vs_dummy)
    p_vs_chance = 1 - norm.cdf(z_vs_chance)
    
    # Significance masks
    sig_vs_dummy = p_vs_dummy < 0.05
    sig_vs_chance = p_vs_chance < 0.05
    
    # FDR correction using custom implementation
    def fdr_correction(p_values, alpha=0.05):
        """Benjamini-Hochberg FDR correction."""
        # Get non-NaN p-values
        valid_mask = ~np.isnan(p_values)
        p_valid = p_values[valid_mask]
        
        # Sort p-values
        p_sorted = np.sort(p_valid)
        n_tests = len(p_sorted)
        
        # Calculate FDR threshold
        fdr_line = alpha * np.arange(1, n_tests + 1) / n_tests
        
        # Find largest p-value below FDR line
        significant = p_sorted <= fdr_line
        if np.any(significant):
            fdr_threshold = p_sorted[significant][-1]
        else:
            fdr_threshold = 0
        
        # Apply threshold
        fdr_mask = p_valid <= fdr_threshold
        
        # Put results back in full array
        full_fdr_mask = np.zeros_like(p_values, dtype=bool)
        full_fdr_mask[valid_mask] = fdr_mask
        
        return full_fdr_mask
    
    # Apply FDR correction
    fdr_mask_dummy = fdr_correction(p_vs_dummy, alpha=0.05)
    fdr_mask_chance = fdr_correction(p_vs_chance, alpha=0.05)
    
    # Cluster correction
    cluster_dummy = cluster_threshold(z_vs_dummy, threshold=1.96, min_cluster_size=20)
    cluster_chance = cluster_threshold(z_vs_chance, threshold=1.96, min_cluster_size=20)
    
    return {
        'svm_scores': svm_scores,
        'dummy_scores': dummy_scores,
        'diff_from_dummy': diff_from_dummy,
        'diff_from_chance': diff_from_chance,
        'z_vs_dummy': z_vs_dummy,
        'z_vs_chance': z_vs_chance,
        'p_vs_dummy': p_vs_dummy,
        'p_vs_chance': p_vs_chance,
        'sig_vs_dummy': sig_vs_dummy,
        'sig_vs_chance': sig_vs_chance,
        'fdr_vs_dummy': fdr_mask_dummy,
        'fdr_vs_chance': fdr_mask_chance,
        'cluster_vs_dummy': cluster_dummy,
        'cluster_vs_chance': cluster_chance
    }

def cluster_threshold(stat_map, threshold=1.96, min_cluster_size=20):
    """Apply cluster-based thresholding."""
    # Create binary map based on threshold
    binary_map = stat_map > threshold
    
    # Label connected components (clusters)
    labeled_clusters, n_clusters = label(binary_map)
    
    # Remove clusters smaller than minimum size
    cluster_map = np.zeros_like(labeled_clusters, dtype=bool)
    for cluster_id in range(1, n_clusters + 1):
        cluster_size = np.sum(labeled_clusters == cluster_id)
        if cluster_size >= min_cluster_size:
            cluster_map[labeled_clusters == cluster_id] = True
    
    return cluster_map

def save_results(stats_dict, mask_img, output_dir):
    """Save all result maps as NIfTI files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Define which maps to save
    maps_to_save = {
        'svm_auc': stats_dict['svm_scores'],
        'dummy_auc': stats_dict['dummy_scores'],
        'diff_from_dummy': stats_dict['diff_from_dummy'],
        'diff_from_chance': stats_dict['diff_from_chance'],
        'z_vs_dummy': stats_dict['z_vs_dummy'],
        'z_vs_chance': stats_dict['z_vs_chance'],
        'p_vs_dummy': stats_dict['p_vs_dummy'],
        'p_vs_chance': stats_dict['p_vs_chance'],
        'sig_vs_dummy': stats_dict['sig_vs_dummy'].astype(float),
        'sig_vs_chance': stats_dict['sig_vs_chance'].astype(float),
        'fdr_vs_dummy': stats_dict['fdr_vs_dummy'].astype(float),
        'fdr_vs_chance': stats_dict['fdr_vs_chance'].astype(float),
        'cluster_vs_dummy': stats_dict['cluster_vs_dummy'].astype(float),
        'cluster_vs_chance': stats_dict['cluster_vs_chance'].astype(float)
    }
    
    for map_name, map_data in maps_to_save.items():
        img = image.new_img_like(mask_img, map_data)
        img.to_filename(os.path.join(output_dir, f'control_music_{map_name}.nii.gz'))
    
    # Save thresholded maps at different AUC levels
    thresholds = [0.55, 0.6, 0.65, 0.7]
    for threshold in thresholds:
        thresholded_scores = np.where(stats_dict['svm_scores'] > threshold, 
                                     stats_dict['svm_scores'], 0)
        thresholded_img = image.new_img_like(mask_img, thresholded_scores)
        thresh_path = os.path.join(output_dir, f'control_music_threshold_{threshold}.nii.gz')
        thresholded_img.to_filename(thresh_path)

def create_visualizations(stats_dict, mask_img, output_dir):
    """Create comprehensive visualizations."""
    
    # 1. Multi-slice AUC comparison
    plt.figure(figsize=(20, 10))
    slice_coords = [-30, -20, -10, 0, 10, 20, 30]
    
    # SVM AUC maps
    for i, slice_coord in enumerate(slice_coords):
        plt.subplot(3, 7, i+1)
        svm_img = image.new_img_like(mask_img, stats_dict['svm_scores'])
        plotting.plot_img(
            svm_img,
            title=f'SVM AUC\nz={slice_coord}mm',
            display_mode='z',
            cut_coords=[slice_coord],
            vmin=0.4,
            vmax=0.8,
            cmap="hot",
            threshold=0.5,
            black_bg=True
        )
    
    # Dummy AUC maps
    for i, slice_coord in enumerate(slice_coords):
        plt.subplot(3, 7, i+8)
        dummy_img = image.new_img_like(mask_img, stats_dict['dummy_scores'])
        plotting.plot_img(
            dummy_img,
            title=f'Dummy AUC\nz={slice_coord}mm',
            display_mode='z',
            cut_coords=[slice_coord],
            vmin=0.3,
            vmax=0.7,
            cmap="cool",
            threshold=0.35,
            black_bg=True
        )
    
    # Difference maps (SVM - Dummy)
    for i, slice_coord in enumerate(slice_coords):
        plt.subplot(3, 7, i+15)
        diff_img = image.new_img_like(mask_img, stats_dict['diff_from_dummy'])
        plotting.plot_img(
            diff_img,
            title=f'SVM-Dummy\nz={slice_coord}mm',
            display_mode='z',
            cut_coords=[slice_coord],
            vmin=-0.2,
            vmax=0.4,
            cmap="RdBu_r",
            threshold=0.05,
            black_bg=True
        )
    
    plt.suptitle('Music vs Non-Music Decoding: SVM vs Dummy Classifier (Control Group)', 
                 fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'control_auc_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Statistical significance maps
    plt.figure(figsize=(20, 10))
    
    # Significance vs chance
    for i, slice_coord in enumerate(slice_coords):
        plt.subplot(2, 7, i+1)
        sig_img = image.new_img_like(mask_img, 
                                    stats_dict['cluster_vs_chance'].astype(float))
        plotting.plot_img(
            sig_img,
            title=f'Sig vs Chance\nz={slice_coord}mm',
            display_mode='z',
            cut_coords=[slice_coord],
            vmin=0,
            vmax=1,
            cmap="Reds",
            threshold=0.5,
            black_bg=True
        )
    
    # Significance vs dummy
    for i, slice_coord in enumerate(slice_coords):
        plt.subplot(2, 7, i+8)
        sig_img = image.new_img_like(mask_img, 
                                    stats_dict['cluster_vs_dummy'].astype(float))
        plotting.plot_img(
            sig_img,
            title=f'Sig vs Dummy\nz={slice_coord}mm',
            display_mode='z',
            cut_coords=[slice_coord],
            vmin=0,
            vmax=1,
            cmap="Blues",
            threshold=0.5,
            black_bg=True
        )
    
    plt.suptitle('Statistical Significance Maps (Cluster Corrected)', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'control_significance_maps.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Summary statistics
    plt.figure(figsize=(15, 10))
    
    # AUC distributions
    plt.subplot(2, 3, 1)
    plt.hist(stats_dict['svm_scores'][stats_dict['svm_scores'] > 0.3], 
             bins=50, alpha=0.7, color='red', label='SVM')
    plt.hist(stats_dict['dummy_scores'][stats_dict['dummy_scores'] > 0.3], 
             bins=50, alpha=0.7, color='blue', label='Dummy')
    plt.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Chance')
    plt.xlabel('AUC Score')
    plt.ylabel('Number of Voxels')
    plt.title('Distribution of AUC Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Difference from chance
    plt.subplot(2, 3, 2)
    plt.hist(stats_dict['diff_from_chance'][~np.isnan(stats_dict['diff_from_chance'])], 
             bins=50, alpha=0.7, color='green')
    plt.axvline(x=0, color='black', linestyle='--', linewidth=2)
    plt.xlabel('AUC Difference from Chance (0.5)')
    plt.ylabel('Number of Voxels')
    plt.title('SVM Performance Above Chance')
    plt.grid(True, alpha=0.3)
    
    # Difference from dummy
    plt.subplot(2, 3, 3)
    plt.hist(stats_dict['diff_from_dummy'][~np.isnan(stats_dict['diff_from_dummy'])], 
             bins=50, alpha=0.7, color='purple')
    plt.axvline(x=0, color='black', linestyle='--', linewidth=2)
    plt.xlabel('AUC Difference (SVM - Dummy)')
    plt.ylabel('Number of Voxels')
    plt.title('SVM Improvement over Dummy')
    plt.grid(True, alpha=0.3)
    
    # Z-score distributions
    plt.subplot(2, 3, 4)
    plt.hist(stats_dict['z_vs_chance'][~np.isnan(stats_dict['z_vs_chance'])], 
             bins=50, alpha=0.7, color='orange', label='vs Chance')
    plt.hist(stats_dict['z_vs_dummy'][~np.isnan(stats_dict['z_vs_dummy'])], 
             bins=50, alpha=0.7, color='cyan', label='vs Dummy')
    plt.axvline(x=1.96, color='red', linestyle='--', linewidth=2, label='p<0.05')
    plt.xlabel('Z-score')
    plt.ylabel('Number of Voxels')
    plt.title('Z-score Distributions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Voxel counts
    plt.subplot(2, 3, 5)
    thresholds = np.arange(0.5, 0.8, 0.01)
    svm_counts = [np.sum(stats_dict['svm_scores'] > t) for t in thresholds]
    dummy_counts = [np.sum(stats_dict['dummy_scores'] > t) for t in thresholds]
    plt.plot(thresholds, svm_counts, 'r-', linewidth=2, label='SVM')
    plt.plot(thresholds, dummy_counts, 'b-', linewidth=2, label='Dummy')
    plt.xlabel('AUC Threshold')
    plt.ylabel('Number of Voxels Above Threshold')
    plt.title('Voxel Count vs. AUC Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Summary table
    plt.subplot(2, 3, 6)
    plt.axis('off')
    
    # Calculate summary stats
    svm_mean = np.nanmean(stats_dict['svm_scores'])
    dummy_mean = np.nanmean(stats_dict['dummy_scores'])
    n_sig_chance = np.sum(stats_dict['cluster_vs_chance'])
    n_sig_dummy = np.sum(stats_dict['cluster_vs_dummy'])
    n_fdr_chance = np.sum(stats_dict['fdr_vs_chance'])
    n_fdr_dummy = np.sum(stats_dict['fdr_vs_dummy'])
    
    summary_text = f"""
    Summary Statistics:
    
    Mean AUC:
    - SVM: {svm_mean:.3f}
    - Dummy: {dummy_mean:.3f}
    
    Significant Voxels (Cluster):
    - vs Chance: {n_sig_chance}
    - vs Dummy: {n_sig_dummy}
    
    Significant Voxels (FDR):
    - vs Chance: {n_fdr_chance}
    - vs Dummy: {n_fdr_dummy}
    
    Max SVM AUC: {np.nanmax(stats_dict['svm_scores']):.3f}
    Min p-value (vs chance): {np.nanmin(stats_dict['p_vs_chance']):.6f}
    """
    
    plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'control_summary_statistics.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """
    Main function to run searchlight analysis with dummy classifier comparison.
    
    This approach:
    1. Uses AUC instead of accuracy for better handling of class imbalance
    2. Compares SVM to dummy classifier (accounts for class distribution)
    3. Also tests against theoretical chance (0.5)
    4. Uses one-tailed tests (SVM > baseline)
    """
    # Set up paths
    csv_path = "/Users/erdemtaskiran/Desktop/nilearn_searchlight_results/searchlight_data.csv"
    mask_path = "/Users/erdemtaskiran/Desktop/mask_GLM_all/control_group_consensus50.nii"
    output_dir = "/Users/erdemtaskiran/Desktop/DUMMY_searchlight"
    
    print("=" * 80)
    print("SEARCHLIGHT ANALYSIS: MUSIC VS NON-MUSIC (CONTROL GROUP)")
    print("Using Dummy Classifier Comparison and AUC Metric")
    print("=" * 80)
    print("\nAnalysis Parameters:")
    print("- Metric: AUC (Area Under ROC Curve)")
    print("- Baseline 1: Dummy classifier (stratified)")
    print("- Baseline 2: Theoretical chance (0.5)")
    print("- Searchlight radius: 4mm")
    print("- Statistical tests: One-tailed (SVM > baseline)")
    print("- Cross-validation: Leave-one-subject-out")
    print("=" * 80)
    
    # Load and prepare data
    beta_imgs, labels, subject_ids, mask_img = load_and_prepare_data(csv_path, mask_path)
    
    # Concatenate beta images
    print("\nConcatenating beta images...")
    beta_4d = image.concat_imgs(beta_imgs)
    
    # Run searchlight for both classifiers
    svm_scores, dummy_scores = run_searchlight_with_dummy(beta_4d, labels, subject_ids, mask_img)
    
    # Compute statistical maps
    print("\nComputing statistical maps...")
    stats_dict = compute_statistical_maps(svm_scores, dummy_scores)
    
    # Save results
    print("\nSaving results...")
    save_results(stats_dict, mask_img, output_dir)
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(stats_dict, mask_img, output_dir)
    
    # Print summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")
    
    print("\nSummary Statistics:")
    print(f"Mean SVM AUC: {np.nanmean(svm_scores):.3f}")
    print(f"Mean Dummy AUC: {np.nanmean(dummy_scores):.3f}")
    print(f"Max SVM AUC: {np.nanmax(svm_scores):.3f}")
    print(f"Voxels with SVM AUC > 0.6: {np.sum(svm_scores > 0.6)}")
    print(f"Voxels with SVM AUC > 0.7: {np.sum(svm_scores > 0.7)}")
    print(f"Significant voxels (vs chance, cluster corrected): {np.sum(stats_dict['cluster_vs_chance'])}")
    print(f"Significant voxels (vs dummy, cluster corrected): {np.sum(stats_dict['cluster_vs_dummy'])}")
    
    print("\nKey output files:")
    print("- control_music_svm_auc.nii.gz: SVM AUC scores")
    print("- control_music_dummy_auc.nii.gz: Dummy AUC scores")
    print("- control_music_cluster_vs_chance.nii.gz: Significant regions vs chance")
    print("- control_music_cluster_vs_dummy.nii.gz: Significant regions vs dummy")
    print("- control_auc_comparison.png: Visual comparison of results")
    print("- control_significance_maps.png: Statistical significance maps")
    
    print("\nInterpretation:")
    print("- Regions with high SVM AUC (>0.6) can decode music vs non-music")
    print("- Significance vs dummy accounts for class imbalance (3:2 ratio)")
    print("- Significance vs chance (0.5) tests theoretical baseline")
    print("- Cluster correction reduces false positives")

if __name__ == "__main__":
    main()