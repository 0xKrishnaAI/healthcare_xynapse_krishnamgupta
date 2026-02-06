#!/usr/bin/env python3
"""
MRI Preprocessing Engine for Neurological Disorder Classification
==================================================================
Task 1: Dataset Preprocessing Pipeline

This script preprocesses T1-weighted MRI brain scans for deep learning-based 
classification of neurological disorders (CN, MCI, AD).

Preprocessing Steps (Research-Backed, ADNI-aligned):
1. N4 Bias Field Correction - Correct intensity non-uniformities
2. Denoising - Reduce noise while preserving edges
3. Skull Stripping - Deep learning-based brain extraction (ANTsPyNet)
4. Registration to MNI152 - Standardize anatomical coordinates
5. Tissue Segmentation - Isolate Grey Matter (GM)
6. Intensity Normalization - Min-max scaling to [0,1]
7. Resampling - Uniform target shape (128, 128, 128)

References:
- ADNI preprocessing pipelines for AD classification
- ANTsPy documentation for registration and segmentation
- ANTsPyNet for deep learning-based brain extraction

Author: AI Engineer | Hackathon 2026
"""

import os
import sys
import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
import ants
import antspynet
from sklearn.model_selection import train_test_split

# =============================================================================
# CONFIGURATION
# =============================================================================

# Directory paths
RAW_DIR = 'data/raw'
PROCESSED_DIR = 'data/processed'
CLINICAL_CSV = 'clinical.csv'

# MNI152 template - Using 2mm resolution for efficiency
# Comment: Using 2mm template to balance anatomical detail with 24-hour 
# processing constraints, per common hackathon optimization.
# Download from: https://github.com/NeuroDesk/neurocontainers or similar
MNI_TEMPLATE = 'MNI152_T1_2mm.nii.gz'

# Target shape for all processed volumes
TARGET_SHAPE = (128, 128, 128)

# Label mapping for classification
LABEL_MAP = {'CN': 0, 'MCI': 1, 'AD': 2}

# Configure logging for errors only
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing_errors.log'),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def center_crop_or_pad(data: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Crop or pad a 3D NumPy array to match the target shape.
    
    This function handles shape mismatches after resampling by:
    - Center cropping if the dimension is larger than target
    - Zero padding if the dimension is smaller than target
    
    Args:
        data: Input 3D NumPy array
        target_shape: Desired output shape (D, H, W)
    
    Returns:
        np.ndarray: Array with exact target_shape dimensions
    """
    result = np.zeros(target_shape, dtype=data.dtype)
    
    # Calculate crop/pad for each dimension
    current_shape = data.shape
    
    # Source and destination slices
    src_slices = []
    dst_slices = []
    
    for i in range(3):
        if current_shape[i] > target_shape[i]:
            # Need to crop - center crop
            start = (current_shape[i] - target_shape[i]) // 2
            end = start + target_shape[i]
            src_slices.append(slice(start, end))
            dst_slices.append(slice(0, target_shape[i]))
        elif current_shape[i] < target_shape[i]:
            # Need to pad - center pad
            start = (target_shape[i] - current_shape[i]) // 2
            end = start + current_shape[i]
            src_slices.append(slice(0, current_shape[i]))
            dst_slices.append(slice(start, end))
        else:
            # Same size
            src_slices.append(slice(0, current_shape[i]))
            dst_slices.append(slice(0, target_shape[i]))
    
    result[tuple(dst_slices)] = data[tuple(src_slices)]
    return result


def validate_template(template_path: str) -> ants.ANTsImage:
    """
    Load and validate the MNI152 template.
    
    Args:
        template_path: Path to the MNI152 template file
    
    Returns:
        ANTsImage: Loaded template
    
    Raises:
        FileNotFoundError: If template doesn't exist
    """
    if not os.path.exists(template_path):
        raise FileNotFoundError(
            f"MNI152 template not found at '{template_path}'. "
            f"Please download the 2mm template from a reliable source."
        )
    return ants.image_read(template_path)


# =============================================================================
# MAIN PREPROCESSING FUNCTION
# =============================================================================

def process_single_mri(input_path: str, template: ants.ANTsImage) -> ants.ANTsImage:
    """
    Process a single T1-weighted MRI scan through the complete preprocessing pipeline.
    
    Pipeline Steps:
    1. Load and initial corrections (N4 bias, denoising)
    2. Stage A: Skull stripping via deep learning
    3. Spatial normalization to MNI152
    4. Stage B: Tissue segmentation and GM isolation
    5. Intensity normalization (min-max to [0,1])
    6. Resampling to target shape
    7. Stage C: Verification
    
    Args:
        input_path: Path to the raw .nii.gz MRI file
        template: Pre-loaded MNI152 template ANTsImage
    
    Returns:
        ants.ANTsImage: Processed grey matter volume
    
    Raises:
        ValueError: If processing fails verification checks
    """
    
    # -------------------------------------------------------------------------
    # Step 1: Load and Initial Corrections
    # -------------------------------------------------------------------------
    # Research Note: N4 bias field correction is standard in ADNI pipelines
    # to correct for intensity non-uniformities caused by RF field inhomogeneity
    
    image = ants.image_read(input_path)
    
    # N4 Bias Field Correction
    # Corrects intensity non-uniformities from MRI scanner
    image = ants.n4_bias_field_correction(image)
    
    # Denoising
    # Reduces noise while preserving edges and anatomical details
    image = ants.denoise_image(image)
    
    # -------------------------------------------------------------------------
    # Step 2 - Stage A: Skull Stripping
    # -------------------------------------------------------------------------
    # Using ANTsPyNet's deep learning-based brain extraction
    # This is more robust than traditional methods like BET
    
    brain_prob = antspynet.brain_extraction(image, modality='t1')
    
    # Threshold probability map at 0.5 to create binary mask
    brain_mask = ants.threshold_image(brain_prob, 0.5, 1.0, 1, 0)
    
    # Apply mask to isolate brain tissue
    brain = image * brain_mask
    
    # -------------------------------------------------------------------------
    # Step 3: Spatial Normalization/Registration to MNI152
    # -------------------------------------------------------------------------
    # Registration to MNI152: To standardize anatomical coordinates across all 
    # subjects to ensure the model learns pathology, not individual head positioning.
    #
    # Using 'SyNOnly' (Symmetric Normalization) transform for accuracy over 'Affine'
    # to better handle non-linear anatomical variations between subjects.
    # SyN is computationally expensive but provides superior registration quality
    # essential for accurate GM comparison across subjects.
    
    registration = ants.registration(
        fixed=template,
        moving=brain,
        type_of_transform='SyNOnly'
    )
    
    registered_brain = registration['warpedmovout']
    
    # -------------------------------------------------------------------------
    # Step 4 - Stage B: Tissue Segmentation (Grey Matter Isolation)
    # -------------------------------------------------------------------------
    # Perform 3-class tissue segmentation: CSF (1), GM (2), WM (3)
    # Grey matter atrophy is a key biomarker for AD progression
    #
    # Parameters explained:
    # - m='[0.2,1x1x1]': MRF parameters (smoothness weight, neighborhood)
    # - c='[5,0]': Convergence parameters (max iterations, threshold)
    # - priorweight=0.8: Weight for prior probability (tissue priors)
    
    # Create a mask for segmentation (threshold above mean intensity)
    brain_array = registered_brain.numpy()
    mean_intensity = np.mean(brain_array[brain_array > 0])
    seg_mask = ants.threshold_image(registered_brain, mean_intensity * 0.1, 1e10, 1, 0)
    
    # Atropos segmentation with 3 classes
    segmentation = ants.atropos(
        a=registered_brain,
        m='[0.2,1x1x1]',
        c='[5,0]',
        i='kmeans[3]',
        x=seg_mask,
        priorweight=0.0  # No spatial priors, use kmeans initialization
    )
    
    # Extract segmentation labels
    seg_image = segmentation['segmentation']
    
    # Isolate Grey Matter (class 2 in 3-class segmentation)
    # Note: Class assignment can vary; GM is typically the middle intensity class
    gm_mask = ants.threshold_image(seg_image, 2, 2, 1, 0)
    
    # Apply GM mask to registered brain
    gm_volume = registered_brain * gm_mask
    
    # -------------------------------------------------------------------------
    # Step 5: Intensity Normalization
    # -------------------------------------------------------------------------
    # Min-max scaling to [0, 1] range
    # This standardizes intensity values across subjects
    
    gm_array = gm_volume.numpy()
    
    min_val = np.min(gm_array)
    max_val = np.max(gm_array)
    
    if max_val > min_val:
        gm_array = (gm_array - min_val) / (max_val - min_val)
    else:
        # Edge case: constant image (shouldn't happen with real data)
        gm_array = np.zeros_like(gm_array)
        logger.warning("Constant intensity image detected during normalization")
    
    # Convert back to ANTsImage
    gm_normalized = ants.from_numpy(
        gm_array,
        origin=gm_volume.origin,
        spacing=gm_volume.spacing,
        direction=gm_volume.direction
    )
    
    # -------------------------------------------------------------------------
    # Step 6: Resizing/Resampling to Target Shape
    # -------------------------------------------------------------------------
    # Resample to uniform target shape for consistent input to neural network
    # Using linear interpolation (interp_type=0) for smooth resampling
    
    gm_resampled = ants.resample_image(
        gm_normalized,
        TARGET_SHAPE,
        use_voxels=True,
        interp_type=0  # Linear interpolation
    )
    
    # Fallback: Ensure exact shape match with center_crop_or_pad
    resampled_array = gm_resampled.numpy()
    
    if resampled_array.shape != TARGET_SHAPE:
        resampled_array = center_crop_or_pad(resampled_array, TARGET_SHAPE)
        gm_resampled = ants.from_numpy(
            resampled_array,
            origin=gm_resampled.origin,
            spacing=gm_resampled.spacing,
            direction=gm_resampled.direction
        )
    
    # -------------------------------------------------------------------------
    # Step 7 - Stage C: Verification
    # -------------------------------------------------------------------------
    # Verify final output meets requirements
    
    final_array = gm_resampled.numpy()
    
    # Check shape
    if final_array.shape != TARGET_SHAPE:
        raise ValueError(
            f"Shape mismatch after processing: {final_array.shape} != {TARGET_SHAPE}"
        )
    
    # Check for all-zeros (indicates failed processing)
    if np.all(final_array == 0):
        raise ValueError("Processed volume is all zeros - processing failed")
    
    # Check for NaN or Inf values
    if np.any(np.isnan(final_array)) or np.any(np.isinf(final_array)):
        raise ValueError("Processed volume contains NaN or Inf values")
    
    return gm_resampled


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution function for the preprocessing pipeline.
    
    Workflow:
    1. Load clinical data and validate labels
    2. Process each MRI with skip logic for existing files
    3. Perform consistency checks on processed data
    4. Split into train/val/test sets (70/15/15)
    5. Save split CSVs for model training
    """
    
    print("=" * 60)
    print("MRI Preprocessing Pipeline for Neurological Disorders")
    print("=" * 60)
    print()
    
    # -------------------------------------------------------------------------
    # Setup and Validation
    # -------------------------------------------------------------------------
    
    # Create processed directory if it doesn't exist
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Load and validate MNI152 template
    print(f"Loading MNI152 template from: {MNI_TEMPLATE}")
    try:
        template = validate_template(MNI_TEMPLATE)
        print(f"  Template shape: {template.shape}")
        print(f"  Template spacing: {template.spacing}")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("\nPlease download the MNI152 2mm template and place it in the project root.")
        print("Recommended source: https://github.com/ANTsX/ANTs/tree/master/Data")
        sys.exit(1)
    
    # Load clinical CSV
    print(f"\nLoading clinical data from: {CLINICAL_CSV}")
    
    if not os.path.exists(CLINICAL_CSV):
        print(f"ERROR: Clinical CSV not found at '{CLINICAL_CSV}'")
        sys.exit(1)
    
    df = pd.read_csv(CLINICAL_CSV)
    print(f"  Total subjects in CSV: {len(df)}")
    
    # Validate required columns
    required_cols = ['subject_id', 'label']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing required columns: {missing_cols}")
        sys.exit(1)
    
    # Map labels to numeric values
    print("\nMapping labels to numeric values:")
    print(f"  {LABEL_MAP}")
    
    # Validate all labels are valid
    invalid_labels = df[~df['label'].isin(LABEL_MAP.keys())]['label'].unique()
    if len(invalid_labels) > 0:
        raise ValueError(f"Invalid labels found in CSV: {invalid_labels}")
    
    df['label_num'] = df['label'].map(LABEL_MAP)
    
    # Print label distribution
    print("\nLabel distribution:")
    for label, count in df['label'].value_counts().items():
        print(f"  {label}: {count} ({count/len(df)*100:.1f}%)")
    
    # -------------------------------------------------------------------------
    # MRI Processing Loop
    # -------------------------------------------------------------------------
    
    print("\n" + "=" * 60)
    print("Processing MRI Scans")
    print("=" * 60 + "\n")
    
    processed_paths = []
    successful = 0
    skipped = 0
    failed = 0
    missing = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing MRIs"):
        subject_id = row['subject_id']
        input_path = os.path.join(RAW_DIR, f"{subject_id}.nii.gz")
        output_path = os.path.join(PROCESSED_DIR, f"{subject_id}_processed.nii.gz")
        
        # Check if input file exists
        if not os.path.exists(input_path):
            logger.warning(f"Input file not found for subject {subject_id}: {input_path}")
            processed_paths.append(None)
            missing += 1
            continue
        
        # Skip if already processed
        if os.path.exists(output_path):
            tqdm.write(f"  Skipping existing: {subject_id}")
            processed_paths.append(output_path)
            skipped += 1
            continue
        
        # Process the MRI
        try:
            processed_image = process_single_mri(input_path, template)
            
            if processed_image is not None:
                # Save processed volume
                ants.image_write(processed_image, output_path)
                processed_paths.append(output_path)
                successful += 1
            else:
                processed_paths.append(None)
                failed += 1
                
        except Exception as e:
            logger.error(f"Failed to process subject {subject_id}: {str(e)}")
            processed_paths.append(None)
            failed += 1
    
    # Add processed paths to dataframe
    df['processed_path'] = processed_paths
    
    # -------------------------------------------------------------------------
    # Processing Summary
    # -------------------------------------------------------------------------
    
    print("\n" + "=" * 60)
    print("Processing Summary")
    print("=" * 60)
    print(f"  Successfully processed: {successful}")
    print(f"  Skipped (existing):     {skipped}")
    print(f"  Failed:                 {failed}")
    print(f"  Missing raw files:      {missing}")
    print(f"  Total with valid paths: {successful + skipped}")
    
    # Drop rows without processed files
    df_valid = df[df['processed_path'].notna()].copy()
    dropped = len(df) - len(df_valid)
    
    if dropped > 0:
        print(f"\n  Dropped {dropped} subjects without valid processed files")
    
    if len(df_valid) == 0:
        print("\nERROR: No valid processed files. Cannot continue.")
        sys.exit(1)
    
    # -------------------------------------------------------------------------
    # Consistency Check
    # -------------------------------------------------------------------------
    
    print("\n" + "=" * 60)
    print("Consistency Check")
    print("=" * 60)
    
    print("\nVerifying all processed volumes have correct shape...")
    
    consistency_errors = []
    
    for idx, row in tqdm(df_valid.iterrows(), total=len(df_valid), desc="Verifying"):
        processed_path = row['processed_path']
        subject_id = row['subject_id']
        
        try:
            img = ants.image_read(processed_path)
            if img.shape != TARGET_SHAPE:
                consistency_errors.append(
                    f"{subject_id}: shape {img.shape} != {TARGET_SHAPE}"
                )
        except Exception as e:
            consistency_errors.append(f"{subject_id}: failed to load - {str(e)}")
    
    if consistency_errors:
        print("\nConsistency errors found:")
        for error in consistency_errors[:10]:  # Show first 10
            print(f"  - {error}")
        if len(consistency_errors) > 10:
            print(f"  ... and {len(consistency_errors) - 10} more")
        raise ValueError(f"Consistency check failed for {len(consistency_errors)} files")
    
    print(f"  All {len(df_valid)} processed volumes verified successfully!")
    
    # -------------------------------------------------------------------------
    # Dataset Splitting
    # -------------------------------------------------------------------------
    
    print("\n" + "=" * 60)
    print("Dataset Splitting (70% Train / 15% Val / 15% Test)")
    print("=" * 60)
    
    # Stratified split: 70% train, 30% temp
    train_df, temp_df = train_test_split(
        df_valid,
        test_size=0.3,
        stratify=df_valid['label_num'],
        random_state=42
    )
    
    # Split temp: 50% val, 50% test (each 15% of total)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df['label_num'],
        random_state=42
    )
    
    # Save split CSVs
    train_df.to_csv('train.csv', index=False)
    val_df.to_csv('val.csv', index=False)
    test_df.to_csv('test.csv', index=False)
    
    print(f"\nDataset split summary:")
    print(f"  Train: {len(train_df)} samples ({len(train_df)/len(df_valid)*100:.1f}%)")
    print(f"  Val:   {len(val_df)} samples ({len(val_df)/len(df_valid)*100:.1f}%)")
    print(f"  Test:  {len(test_df)} samples ({len(test_df)/len(df_valid)*100:.1f}%)")
    
    # Print label distribution per split
    print("\nLabel distribution per split:")
    for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        print(f"\n  {split_name}:")
        for label in ['CN', 'MCI', 'AD']:
            count = len(split_df[split_df['label'] == label])
            print(f"    {label}: {count}")
    
    # -------------------------------------------------------------------------
    # Final Summary
    # -------------------------------------------------------------------------
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print("\nOutput files:")
    print(f"  - Processed MRIs: {PROCESSED_DIR}/")
    print(f"  - train.csv: {len(train_df)} samples")
    print(f"  - val.csv:   {len(val_df)} samples")
    print(f"  - test.csv:  {len(test_df)} samples")
    print("\nReady for model training!")
    print("=" * 60)


if __name__ == '__main__':
    main()
