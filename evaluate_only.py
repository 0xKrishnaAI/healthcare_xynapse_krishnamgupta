from binary_classifier import get_resnet_model, MRIDataset, evaluate_model, DEVICE, filter_and_remap, get_transforms
from torch.utils.data import DataLoader
import torch
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, f1_score
from torchvision import transforms # Needed if loading transforms, though test usually has none

def main():
    print("Loading test data...", flush=True)
    test_df = pd.read_csv('test.csv')
    
    # Use shared logic
    test_binary = filter_and_remap(test_df)
    
    # Ensure order is preserved for tabular report
    # Use 'get_transforms' for validation/test to ensure normalization matches ResNet training
    test_ds = MRIDataset(test_binary, mode='2d', transform=get_transforms('test'))
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)
    
    print(f"Loading model from binary_ad_classifier.pth...", flush=True)
    model = get_resnet_model(num_classes=2)
    # Load model (map_location ensures it loads on CPU if CUDA not available)
    model.load_state_dict(torch.load('binary_ad_classifier.pth', map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    print("Generating predictions...", flush=True)
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    # --- Tabular Report ---
    print("\n" + "="*50, flush=True)
    print("FINAL PREDICTION REPORT", flush=True)
    print("="*50, flush=True)
    
    # Create valid dataframe for report
    report_df = test_binary.copy().reset_index(drop=True)
    report_df['True Label'] = report_df['label'].map({0: 'CN', 1: 'AD'})
    report_df['Predicted Label'] = [ 'CN' if p == 0 else 'AD' for p in all_preds ]
    report_df['Prob AD'] = [f"{p:.4f}" for p in all_probs]
    
    # Select columns
    final_table = report_df[['subject_id', 'True Label', 'Predicted Label', 'Prob AD']]
    
    # Print manually to avoid tabulate dependency
    headers = ["Subject ID", "True", "Pred", "Prob AD"]
    print(f"{headers[0]:<15} {headers[1]:<10} {headers[2]:<10} {headers[3]:<10}", flush=True)
    print("-" * 50, flush=True)
    
    for _, row in final_table.iterrows():
        print(f"{row['subject_id']:<15} {row['True Label']:<10} {row['Predicted Label']:<10} {row['Prob AD']:<10}", flush=True)

    
    # --- Confusion Matrix ---
    cm = confusion_matrix(all_labels, all_preds)
    
    print("\n" + "="*50, flush=True)
    print("CONFUSION MATRIX", flush=True)
    print("="*50, flush=True)
    print("                 Predicted CN    Predicted AD", flush=True)
    print(f"Actual CN        {cm[0][0]:<15} {cm[0][1]}", flush=True)
    print(f"Actual AD        {cm[1][0]:<15} {cm[1][1]}", flush=True)
    print("="*50, flush=True)
    
    # --- Other Metrics for completeness ---
    acc = accuracy_score(all_labels, all_preds)
    print(f"\nOverall Accuracy: {acc:.4f}", flush=True)

if __name__ == "__main__":
    import sys
    try:
        main()
    except Exception as e:
        print(f"CRITICAL ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
