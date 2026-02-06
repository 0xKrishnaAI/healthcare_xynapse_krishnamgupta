import pandas as pd
import os
import glob

def check_counts():
    print("--- CSV Counts ---")
    try:
        train_len = len(pd.read_csv('train.csv'))
        val_len = len(pd.read_csv('val.csv'))
        test_len = len(pd.read_csv('test.csv'))
        print(f"Train CSV: {train_len}")
        print(f"Val CSV:   {val_len}")
        print(f"Test CSV:  {test_len}")
        print(f"Total CSV: {train_len + val_len + test_len}")
    except Exception as e:
        print(f"Error reading CSVs: {e}")

    print("\n--- File Counts ---")
    try:
        nii_files = glob.glob(os.path.join('data', 'processed', '*.nii.gz'))
        print(f"Total .nii.gz files in data/processed: {len(nii_files)}")
        
        # Check for discrepancies
        total_csv_entries = train_len + val_len + test_len
        if len(nii_files) > total_csv_entries:
            print(f"ALERT: Found {len(nii_files) - total_csv_entries} UNUSED images in the directory!")
        elif len(nii_files) < total_csv_entries:
            print(f"ALERT: Missing {total_csv_entries - len(nii_files)} files referenced in CSVs!")
        else:
            print("All files are accounted for in the CSVs.")
            
    except Exception as e:
        print(f"Error checking files: {e}")

if __name__ == "__main__":
    check_counts()
