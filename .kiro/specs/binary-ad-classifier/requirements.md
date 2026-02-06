# Requirements Document

## Introduction

This document specifies the requirements for a binary Alzheimer's Disease (AD) classification system that distinguishes between Cognitively Normal (CN) and Alzheimer's Disease (AD) patients using 3D Convolutional Neural Networks on preprocessed MRI data. The system builds upon an existing preprocessing pipeline (Task 1) and leverages the existing Simple3DCNN model architecture for binary classification with multi-GPU training support.

## Glossary

- **System**: The binary AD classification training and evaluation pipeline
- **MRIDataset**: Custom PyTorch Dataset class for loading preprocessed MRI volumes
- **Simple3DCNN**: Existing 3D convolutional neural network model architecture
- **CN**: Cognitively Normal - patients without cognitive impairment (label 0)
- **AD**: Alzheimer's Disease - patients with Alzheimer's diagnosis (label 1)
- **MCI**: Mild Cognitive Impairment - intermediate stage (label 2, excluded from binary classification)
- **Preprocessed_Volume**: 128x128x128 grey matter volume in .nii.gz format
- **Balanced_Accuracy**: Primary evaluation metric accounting for class imbalance
- **DataParallel**: PyTorch multi-GPU training wrapper
- **Training_Curves**: Visualization of loss and accuracy over epochs

## Requirements

### Requirement 1: Data Loading and Filtering

**User Story:** As a researcher, I want to load only CN and AD cases from the preprocessed dataset, so that I can train a binary classifier without MCI cases.

#### Acceptance Criteria

1. WHEN the System loads train.csv, val.csv, or test.csv, THE System SHALL filter out all rows where label equals 2 (MCI)
2. WHEN the System processes filtered data, THE System SHALL remap labels such that CN becomes 0 and AD becomes 1
3. WHEN the System encounters a missing or corrupted CSV file, THE System SHALL log an error and terminate gracefully
4. THE System SHALL preserve the subject_id, path, and remapped label for each filtered sample

### Requirement 2: MRI Volume Loading

**User Story:** As a researcher, I want to load preprocessed .nii.gz MRI volumes efficiently, so that I can feed them into the neural network for training.

#### Acceptance Criteria

1. THE MRIDataset SHALL load .nii.gz files using nibabel or SimpleITK libraries
2. WHEN a .nii.gz file is loaded, THE MRIDataset SHALL extract the volume data as a numpy array
3. WHEN a .nii.gz file is loaded, THE MRIDataset SHALL convert the volume to a PyTorch tensor with shape (1, 128, 128, 128)
4. IF a .nii.gz file fails to load, THEN THE MRIDataset SHALL log the error with subject_id and path, and skip that sample
5. THE MRIDataset SHALL normalize volume intensities to have zero mean and unit variance
6. THE MRIDataset SHALL return tuples of (volume_tensor, label) when indexed

### Requirement 3: Multi-GPU Training Configuration

**User Story:** As a researcher, I want to utilize multiple GPUs for training, so that I can accelerate model training within the hackathon time constraint.

#### Acceptance Criteria

1. WHEN 2 or more GPUs are available, THE System SHALL wrap the Simple3DCNN model with torch.nn.DataParallel
2. WHEN only 1 GPU is available, THE System SHALL use that single GPU without DataParallel
3. WHEN no GPUs are available, THE System SHALL fall back to CPU training and log a warning
4. THE System SHALL use a batch size of 4 for training
5. THE System SHALL configure DataLoader with num_workers=4 for parallel data loading
6. WHEN creating the training DataLoader, THE System SHALL enable shuffling
7. WHEN creating validation or test DataLoaders, THE System SHALL disable shuffling

### Requirement 4: Model Training Loop

**User Story:** As a researcher, I want to train the 3D CNN model with proper optimization and validation tracking, so that I can develop an accurate AD classifier.

#### Acceptance Criteria

1. THE System SHALL train the Simple3DCNN model for 20 epochs
2. THE System SHALL use Adam optimizer with learning rate 1e-4
3. THE System SHALL use CrossEntropyLoss as the loss function
4. WHEN training each epoch, THE System SHALL iterate through all training batches with tqdm progress tracking
5. WHEN training each batch, THE System SHALL perform forward pass, loss computation, backward pass, and optimizer step
6. WHEN each epoch completes, THE System SHALL evaluate on the validation set and compute validation loss and accuracy
7. WHEN each epoch completes, THE System SHALL record training loss, training accuracy, validation loss, and validation accuracy
8. IF validation accuracy improves, THEN THE System SHALL save the model checkpoint as 'binary_ad_classifier.pth'
9. IF an error occurs during training, THEN THE System SHALL log the error and attempt to continue or terminate gracefully

### Requirement 5: Training Visualization

**User Story:** As a researcher, I want to visualize training progress, so that I can assess model convergence and detect overfitting.

#### Acceptance Criteria

1. WHEN training completes, THE System SHALL generate a training curves plot with 2 subplots
2. THE System SHALL plot training loss and validation loss in the first subplot
3. THE System SHALL plot training accuracy and validation accuracy in the second subplot
4. THE System SHALL label axes appropriately with "Epoch", "Loss", and "Accuracy"
5. THE System SHALL include a legend distinguishing training and validation curves
6. THE System SHALL save the plot as 'training_curves.png'

### Requirement 6: Model Evaluation

**User Story:** As a researcher, I want comprehensive evaluation metrics on the test set, so that I can assess the clinical viability of the classifier.

#### Acceptance Criteria

1. WHEN evaluation begins, THE System SHALL load the best saved model checkpoint
2. THE System SHALL evaluate the model on the test set without gradient computation
3. THE System SHALL compute Balanced Accuracy as the primary metric
4. THE System SHALL compute AUC-ROC score using predicted probabilities
5. THE System SHALL compute Macro F1-Score
6. THE System SHALL compute Precision for both CN and AD classes
7. THE System SHALL compute Recall for both CN and AD classes
8. THE System SHALL generate a confusion matrix
9. THE System SHALL print all metrics formatted to 4 decimal places
10. IF Balanced Accuracy exceeds 91%, THEN THE System SHALL print a success message indicating the target was achieved

### Requirement 7: Error Handling and Logging

**User Story:** As a researcher, I want robust error handling throughout the pipeline, so that I can diagnose issues without the system crashing.

#### Acceptance Criteria

1. THE System SHALL log all file loading errors with subject_id and file path
2. THE System SHALL log GPU availability and configuration at startup
3. THE System SHALL log training progress including epoch number, losses, and accuracies
4. IF a CUDA out-of-memory error occurs, THEN THE System SHALL log the error with guidance to reduce batch size
5. THE System SHALL write logs to a file named 'training.log'
6. WHEN an unexpected error occurs, THE System SHALL log the full traceback before terminating

### Requirement 8: Output Artifacts

**User Story:** As a researcher, I want all training outputs saved to disk, so that I can reproduce results and deploy the trained model.

#### Acceptance Criteria

1. THE System SHALL save the best model checkpoint as 'binary_ad_classifier.pth'
2. THE System SHALL save the training curves visualization as 'training_curves.png'
3. THE System SHALL save training logs to 'training.log'
4. WHEN saving the model checkpoint, THE System SHALL include the model state_dict
5. THE System SHALL print the file paths of all saved artifacts upon completion
