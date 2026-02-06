from multi_classifier import MRIDataset, Simple3DCNN, evaluate_model, DEVICE, BATCH_SIZE, NUM_WORKERS
from torch.utils.data import DataLoader
import torch
import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.ERROR)

def main():
    print("Loading Multi-Class Test Data...", flush=True)
    test_df = pd.read_csv('test.csv')
    test_ds = MRIDataset(test_df)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False) # Reduced batch size for safety

    print("Loading Multi-Class Model (multi_ad_classifier.pth)...", flush=True)
    model = Simple3DCNN(num_classes=3)
    try:
        model.load_state_dict(torch.load('multi_ad_classifier.pth', map_location=DEVICE))
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.to(DEVICE)
    
    print("Running Evaluation...", flush=True)
    evaluate_model(model, test_loader)

if __name__ == "__main__":
    main()
