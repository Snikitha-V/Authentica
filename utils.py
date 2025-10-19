import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

def compute_metrics(labels, distances, threshold=0.5):
    labels = np.array(labels)
    distances = np.array(distances)
    pred_labels = (distances < threshold).astype(int)

    cm = confusion_matrix(labels, pred_labels)
    if cm.shape == (2,2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    FAR = fp / (fp + tn) if (fp + tn) > 0 else 0
    FRR = fn / (fn + tp) if (fn + tp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    try:
        auc = roc_auc_score(labels, -distances)
    except:
        auc = 0.0

    return {"accuracy": accuracy, "AUC": auc, "FAR": FAR, "FRR": FRR, "confusion_matrix": cm}
