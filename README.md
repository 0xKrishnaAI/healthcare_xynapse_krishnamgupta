# 🧠 MRI Preprocessing Pipeline for Alzheimer's Classification

Deep learning-ready preprocessing for T1-weighted MRI brain scans to classify **CN** (Cognitively Normal), **MCI** (Mild Cognitive Impairment), and **AD** (Alzheimer's Disease).

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/hackathon.git
cd hackathon

# 2. Install dependencies
pip install -r requirements.txt

# 3. Prepare your data
#    - Place MRI scans (.nii.gz) in data/raw/
#    - Create clinical.csv with subject_id and label columns (see clinical_example.csv)

# 4. Run preprocessing
python preprocess_engine.py
```

## 📁 Project Structure

```
hackathon/
├── preprocess_engine.py    # Main preprocessing pipeline (576 lines)
├── convert_dicom_to_nifti.py  # DICOM to NIfTI converter (optional)
├── requirements.txt        # Python dependencies
├── clinical_example.csv    # Example input format
├── MNI152_T1_2mm.nii.gz   # MNI152 template for registration
└── data/
    ├── raw/               # Place your .nii.gz MRI scans here
    └── processed/         # Preprocessed outputs (auto-generated)
```

## 🔬 Preprocessing Pipeline

| Step | Method | Purpose |
|------|--------|---------|
| 1 | N4 Bias Correction | Remove intensity non-uniformities |
| 2 | Denoising | Reduce noise while preserving edges |
| 3 | Skull Stripping | Deep learning brain extraction (ANTsPyNet) |
| 4 | MNI152 Registration | Standardize anatomical coordinates (SyNOnly) |
| 5 | Tissue Segmentation | 3-class Atropos (CSF, GM, WM) |
| 6 | Grey Matter Isolation | Extract GM for AD biomarker analysis |
| 7 | Intensity Normalization | Min-max scaling to [0,1] |
| 8 | Resampling | Uniform 128×128×128 voxels |

## 📊 Input Format

**clinical.csv** (required):
```csv
subject_id,label
136_S_0300,AD
136_S_0196,CN
136_S_0579,MCI
```

**MRI files**: Place as `data/raw/{subject_id}.nii.gz`

## 📤 Output

- `data/processed/*_processed.nii.gz` — Preprocessed GM volumes
- `train.csv`, `val.csv`, `test.csv` — 70/15/15 stratified splits

## ⚙️ Features

- ✅ **Skip Logic**: Resumes from where it left off
- ✅ **Progress Bars**: Real-time tqdm tracking
- ✅ **Error Logging**: Detailed logs in `preprocessing_errors.log`
- ✅ **Consistency Checks**: Verifies output shapes and values
- ✅ **Research-Backed**: ADNI-aligned preprocessing steps

## 📋 Requirements

- Python 3.8+
- ~4GB RAM per MRI scan
- GPU optional (speeds up skull stripping)

## 📚 References

- ADNI Preprocessing Protocols
- ANTsPy/ANTsPyNet Documentation
- MNI152 Standard Template

---

**Built for Hackathon 2026** 🏆
