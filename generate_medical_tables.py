import pandas as pd
import torch
import torch.nn as nn
import nibabel as nib
import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix

# Load test data
test_df = pd.read_csv('test.csv')

# Define model architecture (same as training)
class Simple3DCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(Simple3DCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * 16 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Load model
device = torch.device('cpu')
model = Simple3DCNN(num_classes=3)
model.load_state_dict(torch.load('multi_ad_classifier.pth', map_location=device))
model.eval()

# Run predictions
all_subject_ids = []
all_preds = []
all_labels = []
all_probs = []

print("Running predictions on test set...")
for idx, row in test_df.iterrows():
    subject_id = row['subject_id']
    img_path = row['path']
    label = int(row['label'])
    
    # Load image
    img = nib.load(img_path)
    img_np = img.get_fdata().astype(np.float32)
    img_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    
    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)[0].numpy()
        pred = int(torch.argmax(output, dim=1).item())
    
    all_subject_ids.append(subject_id)
    all_preds.append(pred)
    all_labels.append(label)
    all_probs.append(probs)

# Convert to numpy arrays
all_probs = np.array(all_probs)

# Calculate metrics
bal_acc = balanced_accuracy_score(all_labels, all_preds)
try:
    auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
except:
    auc = 0.0
f1 = f1_score(all_labels, all_preds, average='macro')
precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
cm = confusion_matrix(all_labels, all_preds)

# Map labels to class names
class_names = {0: 'CN', 1: 'MCI', 2: 'AD'}

# Generate results dictionary for table generation
results = {
    'subject_ids': all_subject_ids,
    'predictions': [class_names[p] for p in all_preds],
    'confidences': [max(probs) * 100 for probs in all_probs],
    'actual_labels': [class_names[l] for l in all_labels],
    'balanced_accuracy': bal_acc,
    'auc': auc,
    'f1': f1,
    'precision_cn': precision[0],
    'precision_mci': precision[1],
    'precision_ad': precision[2],
    'recall_cn': recall[0],
    'recall_mci': recall[1],
    'recall_ad': recall[2],
    'confusion_matrix': cm
}

# Save results
import pickle
with open('multi_test_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print(f"\\nEvaluation complete!")
print(f"Balanced Accuracy: {bal_acc:.4f}")
print(f"AUC (Macro OvR): {auc:.4f}")
print(f"Macro F1-Score: {f1:.4f}")
print(f"\\nResults saved to 'multi_test_results.pkl'")
