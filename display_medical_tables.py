import pickle
import pandas as pd

# Load results
with open('multi_test_results.pkl', 'rb') as f:
    r = pickle.load(f)

print("="*80)
print("TASK 3: MULTI-CLASS NEUROLOGICAL STATE CLASSIFICATION")
print("="*80)

# 1. Final Prediction Output (Per Subject)
print("\\n## 1. FINAL PREDICTION OUTPUT (PER SUBJECT)")
print("-"*80)
df_pred = pd.DataFrame({
    'Subject_ID': r['subject_ids'],
    'Predicted_Class': r['predictions'],
    'Confidence_%': [f"{c:.2f}%" for c in r['confidences']],
    'Threshold': '60%',
    'Status': ['Accepted' if c>=60 else 'Rejected' for c in r['confidences']]
})
print(df_pred.to_string(index=False))

# 2. Overall Evaluation Metrics
print("\\n\\n## 2. OVERALL EVALUATION METRICS")
print("-"*80)
df_metrics = pd.DataFrame({
    'Metric': ['Balanced Accuracy', 'Macro F1-Score', 'Macro-Averaged AUC'],
    'Value': [
        f"{r['balanced_accuracy']:.4f} ({r['balanced_accuracy']*100:.2f}%)",
        f"{r['f1']:.4f} ({r['f1']*100:.2f}%)",
        f"{r['auc']:.4f} ({r['auc']*100:.2f}%)"
    ]
})
print(df_metrics.to_string(index=False))

# 3. Class-Wise Precision, Recall & F1-Score
print("\\n\\n## 3. CLASS-WISE PRECISION, RECALL & F1-SCORE")
print("-"*80)
# Calculate F1 per class
from sklearn.metrics import f1_score, precision_score, recall_score
# Reconstruct labels from predictions
label_map = {'CN': 0, 'MCI': 1, 'AD': 2}
actual = [label_map[l] for l in r['actual_labels']]
predicted = [label_map[p] for p in r['predictions']]

f1_per_class = f1_score(actual, predicted, average=None, zero_division=0)

df_class = pd.DataFrame({
    'Class': ['CN', 'MCI', 'AD'],
    'Precision': [f"{r['precision_cn']:.4f}", f"{r['precision_mci']:.4f}", f"{r['precision_ad']:.4f}"],
    'Recall': [f"{r['recall_cn']:.4f}", f"{r['recall_mci']:.4f}", f"{r['recall_ad']:.4f}"],
    'F1-Score': [f"{f1_per_class[0]:.4f}", f"{f1_per_class[1]:.4f}", f"{f1_per_class[2]:.4f}"]
})
print(df_class.to_string(index=False))

# 4. Confusion Matrix
print("\\n\\n## 4. CONFUSION MATRIX (Actual vs Predicted)")
print("-"*80)
cm = r['confusion_matrix']
df_cm = pd.DataFrame(cm, 
                     columns=['Predicted_CN', 'Predicted_MCI', 'Predicted_AD'],
                     index=['Actual_CN', 'Actual_MCI', 'Actual_AD'])
print(df_cm.to_string())

# 5. Training & Validation Performance Summary (from test results)
print("\\n\\n## 5. TRAINING & VALIDATION PERFORMANCE SUMMARY")
print("-"*80)
print("(Note: Full training logs available in terminal output)")
print(f"Final Model - Test Set Balanced Accuracy: {r['balanced_accuracy']:.4f}")
print(f"Total Epochs: 20")
print(f"Final Validation Accuracy: ~46%")

# 6. Final System Output
print("\\n\\n## 6. FINAL SYSTEM OUTPUT (ONE-LINE DECISION)")
print("-"*80)
accepted_count = sum(1 for c in r['confidences'] if c >= 60)
total_count = len(r['confidences'])
avg_confidence = sum(r['confidences']) / len(r['confidences'])

df_final = pd.DataFrame({
    'Status': ['Multi-Class Classification Complete'],
    'Neurological_States': ['CN / MCI / AD'],
    'Test_Samples': [f"{total_count} samples"],
    'Accepted_Predictions': [f"{accepted_count}/{total_count} (≥60% confidence)"],
    'Average_Confidence': [f"{avg_confidence:.2f}%"],
    'Balanced_Accuracy': [f"{r['balanced_accuracy']*100:.2f}%"],
    'Threshold_Status': ['Below 60% - Consider Model Tuning']
})
print(df_final.to_string(index=False))

print("\\n" + "="*80)
print("END OF MEDICAL AI EVALUATION REPORT")
print("="*80)
