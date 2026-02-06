import os
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
BATCH_SIZE = 8 # Increased slightly for 2D which uses less memory
NUM_EPOCHS = 250 # Increased to 250 as requested (50 + 200)
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 0 # Safe for Windows. adjust if needed.

# Helper function defined globally to be importable
def filter_and_remap(df):
    """
    Filters DataFrame for CN (0) and AD (2) classes and remaps AD to 1.
    """
    # Check if 'label' column exists
    if 'label' not in df.columns:
       # Fallback or error if structure is unexpected
       raise ValueError("DataFrame missing required 'label' column")
    
    df_binary = df[df['label'].isin([0, 2])].copy()
    # Remap: 0->0 (CN), 2->1 (AD)
    df_binary['label'] = df_binary['label'].map({0: 0, 2: 1})
    # Also ensure 'label_num' exists for compatibility
    df_binary['label_num'] = df_binary['label'] 
    return df_binary

def get_transforms(phase):
    """
    Returns transforms for data augmentation.
    """
    if phase == 'train':
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            # transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)), # Optional: slight shift
        ])
    else:
        return None # No transforms for validation/test

class MRIDataset(Dataset):
    """
    Dataset class for loading MRI volumes. 
    Supports both 3D volumes and 2D middle-slice extraction.
    """
    def __init__(self, metadata_df, mode='3d', transform=None):
        self.metadata = metadata_df
        self.mode = mode.lower()
        self.transform = transform
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_path = row['path'] # The CSV header is 'path', not 'processed_path'
        # Safety cast: Handle floats or strings that look like floats
        try:
            label = int(float(row['label'])) 
        except ValueError:
            label = int(row['label']) # Fallback
        
        try:
            # Load image using nibabel
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
                
            img = nib.load(img_path)
            img_np = img.get_fdata().astype(np.float32) # Expected shape (128, 128, 128)
            
            if self.mode == '3d':
                # Add channel dimension: (1, 128, 128, 128)
                img_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0)
            elif self.mode == '2d':
                # Extract middle slice (axial)
                # Assuming shape (128, 128, 128), middle is index 64 along axis 2
                mid_slice = img_np[:, :, img_np.shape[2] // 2]
                # Add channel dimension: (1, 128, 128)
                img_tensor = torch.tensor(mid_slice, dtype=torch.float32).unsqueeze(0)
            else:
                raise ValueError("Mode must be '2d' or '3d'")
            
            # Apply transforms if 2d (transforms usually expect PIL or Tensor)
            # Our tensor is (C, H, W). torchvision transforms work on this.
            if self.mode == '2d' and self.transform:
                img_tensor = self.transform(img_tensor)
                
            return img_tensor, torch.tensor(label, dtype=torch.long)
            
        except Exception as e:
            logger.error(f"Error loading {img_path}: {str(e)}")
            # Return zero tensor in case of error to prevent crash
            if self.mode == '3d':
                return torch.zeros((1, 128, 128, 128)), torch.tensor(label, dtype=torch.long)
            else:
                return torch.zeros((1, 128, 128)), torch.tensor(label, dtype=torch.long)

class Simple3DCNN(nn.Module):
    """
    Simple 3D CNN for AD classification, research-backed by ADNI studies on structural MRI for binary CN/AD tasks.
    """
    def __init__(self):
        super(Simple3DCNN, self).__init__()
        self.features = nn.Sequential(
            # Conv Layer 1: 1 -> 32
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            # Conv Layer 2: 32 -> 64
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            # Conv Layer 3: 64 -> 128
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            # Input size calculation: 128 -> 64 -> 32 -> 16. 
            # 16*16*16 * 128 = 524288 parameters. This might be large.
            # Let's add an AdaptiveAvgPool to reduce size before dense layers if needed, 
            # but following the prompt: flattened to 512.
            # 128 * 16 * 16 * 16 = 524288
            nn.Linear(128 * 16 * 16 * 16, 512), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class Simple2DCNN(nn.Module):
    """
    Simple 2D CNN extracting features from the middle axial slice. 
    Often used as a lightweight baseline or complementary approach.
    """
    def __init__(self):
        super(Simple2DCNN, self).__init__()
        self.features = nn.Sequential(
            # Conv Layer 1: 1 -> 32
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), # Added Batch Norm
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv Layer 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), # Added Batch Norm
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv Layer 3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), # Added Batch Norm
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            # 128 * 16 * 16
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_model(model, train_loader, val_loader, model_name="model"):
    model = model.to(DEVICE)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training.")
        model = nn.DataParallel(model)
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5) # Added weight decay (L2 regularization)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1) # Decay LR every 15 epochs
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    print(f"\nStarting training for {model_name} on {DEVICE}...")
    
    for epoch in range(NUM_EPOCHS):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]", leave=False)
        for images, labels in loop:
            try:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                loop.set_postfix(loss=loss.item())
            except Exception as e:
                logger.error(f"Error in training step: {e}")
                continue
        
        # Step the scheduler
        scheduler.step()
                
        epoch_loss = running_loss / total if total > 0 else 0
        epoch_acc = correct / total if total > 0 else 0
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        # Validation Phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                try:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_running_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                except Exception as e:
                    logger.error(f"Error in validation step: {e}")
                    continue
        
        val_loss = val_running_loss / val_total if val_total > 0 else 0
        val_acc = val_correct / val_total if val_total > 0 else 0
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}: Train Loss={epoch_loss:.4f}, Train Acc={epoch_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title(f'{model_name} Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.legend()
    plt.title(f'{model_name} Accuracy')
    
    plt.savefig(f'{model_name}_training_curves.png')
    plt.close()
    
    return model

def evaluate_model(model, test_loader, model_name="model"):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    print(f"\nEvaluating {model_name}...")
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            try:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)[:, 1] # Probability of AD (class 1)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
            except Exception as e:
                logger.error(f"Error in evaluation step: {e}")
                
    # Metrics
    if len(all_labels) == 0:
        print("No data evaluated.")
        return

    bal_acc = accuracy_score(all_labels, all_preds) # Balanced if dataset is balanced, but prompt asks for 'Balanced Accuracy' specifically
    # Actually sklearn has balanced_accuracy_score
    from sklearn.metrics import balanced_accuracy_score
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0 # Handle case with only one class
        
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average=None)
    recall = recall_score(all_labels, all_preds, average=None)
    cm = confusion_matrix(all_labels, all_preds)
    
    print(f"\nResults for {model_name}:")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Macro F1-Score: {f1:.4f}")
    print(f"Precision (CN, AD): {precision}")
    print(f"Recall (CN, AD): {recall}")
    print(f"Confusion Matrix:\n{cm}")
    
    if bal_acc > 0.91:
        print("SUCCESS: Balanced Accuracy exceeds 91%!")
    else:
        print("NOTE: Balanced Accuracy is below 91%. Consider tuning hyperparameters, increasing epochs, or checking data quality.")

def main():
    try:
        # 1. Load DataFrames
        train_df = pd.read_csv('train.csv')
        val_df = pd.read_csv('val.csv')
        test_df = pd.read_csv('test.csv')
        
        # 2. Filter for CN (0) and AD (2) only
        # The CSV contains 'label' column with 0, 1, 2.
        # We need to filter out MCI (1) and remap AD (2) to 1.
        
        # Helper function is now global
        train_binary = filter_and_remap(train_df)
        val_binary = filter_and_remap(val_df)
        test_binary = filter_and_remap(test_df)
        
        print(f"Binary Dataset Sizes: Train={len(train_binary)}, Val={len(val_binary)}, Test={len(test_binary)}")
        
        # ---------------------------------------------------------
        # 3D Model Execution (SKIPPED/REMOVED as per user request to pivot to 2D)
        # ---------------------------------------------------------
        # Legacy 3D code removed for cleanliness. 
        # Refer to git history or separate file if needed.
        
        # ---------------------------------------------------------
        # 2D Model Execution (PRIMARY)
        # ---------------------------------------------------------
        print("\n" + "="*30)
        print("Running 2D CNN Pipeline (PRIMARY)")
        print("="*30)
        
        # Apply transforms only to training set
        train_ds_2d = MRIDataset(train_binary, mode='2d', transform=get_transforms('train'))
        val_ds_2d = MRIDataset(val_binary, mode='2d', transform=get_transforms('val'))
        test_ds_2d = MRIDataset(test_binary, mode='2d', transform=get_transforms('test'))
        
        train_loader_2d = DataLoader(train_ds_2d, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_loader_2d = DataLoader(val_ds_2d, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        test_loader_2d = DataLoader(test_ds_2d, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        
        model_2d = Simple2DCNN()
        model_2d = train_model(model_2d, train_loader_2d, val_loader_2d, model_name="2D_CNN")
        
        # Save 2D model as the primary output
        torch.save(model_2d.state_dict(), 'binary_ad_classifier.pth')
        print("Saved 2D model as 'binary_ad_classifier.pth'")
        
        evaluate_model(model_2d, test_loader_2d, model_name="2D_CNN")
        
        print("\nTraining and evaluation complete. (2D Model Selected)")
        
    except Exception as e:
        logger.critical(f"Critical failure in main execution: {e}", exc_info=True)

if __name__ == '__main__':
    main()
