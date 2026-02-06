import torch
import torch.nn as nn
import torch.nn.functional as F

class Simple3DCNN(nn.Module):
    """
    A lightweight 3D CNN for MRI Classification.
    Input: ( Batch, 1, 128, 128, 128 )
    Output: ( Batch, 3 ) -> [CN, MCI, AD]
    """
    def __init__(self, num_classes=3):
        super(Simple3DCNN, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2) # 64x64x64
        
        # Block 2
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2) # 32x32x32
        
        # Block 3
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2) # 16x16x16
        
        # Block 4
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(256)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2) # 8x8x8
        
        # Global Average Pooling (Reduce to 1x1x1)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # Classification Head
        self.fc1 = nn.Linear(256, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # x shape: (B, 1, 128, 128, 128)
        
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # x shape: (B, 256, 8, 8, 8) -> Pool -> (B, 256, 1, 1, 1)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1) # Flatten -> (B, 256)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x) # Output logits
        
        return x

if __name__ == "__main__":
    # Test the model structure
    model = Simple3DCNN(num_classes=3)
    print(model)
    
    # Dummy input check
    dummy_input = torch.randn(1, 1, 128, 128, 128)
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape} (Should be [1, 3])")
