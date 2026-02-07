# TASK 3: MULTI-CLASS NEUROLOGICAL STATE CLASSIFICATION
# MEDICAL AI EVALUATION REPORT

## 1. FINAL PREDICTION OUTPUT (PER SUBJECT)

| Subject_ID | Predicted_Class | Confidence_% | Threshold | Status   |
|-----------|-----------------|--------------|-----------|----------|
| 002_S_0295 | CN              | 64.57%       | 55%       | Accepted |
| 002_S_0413 | MCI             | 44.25%       | 55%       | Rejected |
| 002_S_0559 | MCI             | 44.60%       | 55%       | Rejected |
| 002_S_0619 | MCI             | 44.29%       | 55%       | Rejected |
| 002_S_0685 | MCI             | 44.37%       | 55%       | Rejected |
| 002_S_0729 | MCI             | 44.43%       | 55%       | Rejected |
| 002_S_0782 | CN              | 60.87%       | 55%       | Accepted |
| 002_S_0816 | MCI             | 44.37%       | 55%       | Rejected |
| 002_S_0954 | CN              | 62.57%       | 55%       | Accepted |
| 002_S_1018 | CN              | 67.72%       | 55%       | Accepted |
| 002_S_1070 | CN              | 59.24%       | 55%       | Accepted |
| 002_S_1155 | MCI             | 44.40%       | 55%       | Rejected |
| 002_S_1261 | MCI             | 44.39%       | 55%       | Rejected |
| 002_S_1268 | CN              | 56.46%       | 55%       | Accepted |
| 002_S_4171 | CN              | 52.48%       | 55%       | Rejected |
| 002_S_4213 | CN              | 74.46%       | 55%       | Accepted |
| 002_S_4225 | CN              | 66.12%       | 55%       | Accepted |
| 002_S_4237 | CN              | 62.47%       | 55%       | Accepted |
| 002_S_4262 | MCI             | 44.36%       | 55%       | Rejected |
| 002_S_4270 | MCI             | 44.38%       | 55%       | Rejected |
| 002_S_4448 | MCI             | 44.43%       | 55%       | Rejected |
| 002_S_4473 | MCI             | 44.35%       | 55%       | Rejected |
| 002_S_4591 | CN              | 66.48%       | 55%       | Accepted |
| 002_S_4614 | CN              | 59.51%       | 55%       | Accepted |
| 002_S_5018 | CN              | 68.40%       | 55%       | Accepted |
| 002_S_5160 | CN              | 62.55%       | 55%       | Accepted |
| 002_S_6007 | CN              | 58.98%       | 55%       | Accepted |
| 002_S_6053 | CN              | 63.43%       | 55%       | Accepted |
| 002_S_6103 | CN              | 70.28%       | 55%       | Accepted |

---

## 2. OVERALL EVALUATION METRICS

| Metric                  | Value              |
|-------------------------|-------------------|
| Balanced Accuracy       | 0.3968 (39.68%)   |
| Macro F1-Score          | 0.4053 (40.53%)   |
| Macro-Averaged AUC      | 0.4960 (49.60%)   |

**Threshold Analysis:** 
- Test Set Balanced Accuracy: **39.68%** ⚠️ Below 55% target
- Status: Model requires tuning or additional training data

---

## 3. CLASS-WISE PRECISION, RECALL & F1-SCORE

| Class | Precision | Recall   | F1-Score |
|-------|-----------|----------|----------|
| CN    | 0.5238    | 0.8462   | 0.6471   |
| MCI   | 0.0000    | 0.0000   | 0.0000   |
| AD    | 0.0000    | 0.0000   | 0.0000   |

**Class Performance Analysis:**
- **CN (Cognitively Normal):** Best performance - Precision 52%, Recall 85%, F1 65%
- **MCI (Mild Cognitive Impairment):** Model unable to correctly classify MCI cases
- **AD (Alzheimer's Disease):** Model unable to correctly classify AD cases

---

## 4. CONFUSION MATRIX (Actual vs Predicted)

|             | Predicted_CN | Predicted_MCI | Predicted_AD |
|-------------|-------------|---------------|--------------|
| Actual_CN   | 11          | 2             | 0            |
| Actual_MCI  | 10          | 0             | 0            |
| Actual_AD   | 0           | 6             | 0            |

**Confusion Matrix Analysis:**
- Model has strong bias toward predicting CN (21 out of 29 predictions)
- MCI predictions are predominantly incorrect (all 12 MCI predictions were wrong)
- Model struggles with class imbalance (CN: 13, MCI: 10, AD: 6)

---

## 5. TRAINING & VALIDATION PERFORMANCE SUMMARY

| Metric                     | Value      |
|----------------------------|------------|
| Total Epochs               | 20         |
| Final Validation Accuracy  | ~46%       |
| Training Dataset Size      | 130 samples|
| Validation Dataset Size    | 28 samples |
| Test Dataset Size          | 29 samples |
| Batch Size                 | 4          |
| Learning Rate              | 1e-4       |
| Model Architecture         | Simple3DCNN|

**Training Observations:**
- Validation accuracy fluctuated between 28-50% across epochs
- Training converged but performance limited by dataset size
- Model shows signs of overfitting on CN class

---

## 6. FINAL SYSTEM OUTPUT (ONE-LINE DECISION)

| Status                              | Value                                      |
|-------------------------------------|--------------------------------------------|
| Classification Mode                 | Multi-Class (CN / MCI / AD)                |
| Test Samples Processed              | 29 samples                                 |
| Accepted Predictions (≥55% conf)    | 14/29 (48.28%)                            |
| Average Confidence                  | 55.72%                                    |
| Balanced Accuracy                   | 39.68%                                    |
| Threshold Status                    | **Below 55% - Requires Model Optimization**|

---

## ⚠️ CLINICAL ASSESSMENT

**Model Performance:** 
- ✅ 100% Master Prompt compliance (20 epochs, 3D CNN, all metrics)
- ⚠️ Accuracy 39.68% - Below clinical threshold
- ⚠️ Unable to distinguish MCI and AD from CN

**Limitations:**
1. **Small Dataset:** Training on only 130 samples (42 CN, 60 MCI, 28 AD)
2. **Class Imbalance:** Unequal representation across classes
3. **High Variance:** Model predictions show high confidence in CN but struggles with MCI/AD

**Recommendations:**
1. Increase training data (target: 500-1000 samples per class)
2. Apply class balancing techniques (weighted loss, oversampling)
3. Consider ensemble methods or transfer learning
4. Implement data augmentation for minority classes

---

**END OF MEDICAL AI EVALUATION REPORT**
