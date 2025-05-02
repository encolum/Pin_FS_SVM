import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score, confusion_matrix, f1_score

def evaluate_model(y_true, y_pred):
    """
    Evaluate model performance using multiple metrics
    
    Parameters:
    -----------
    y_true : numpy array
        True labels
    y_pred : numpy array
        Predicted labels
    Returns
    --------
    dict
        Dictionary containing evaluation metrics
    """
        
    acc = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    # Calculate F1-score
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Handle binary AUC calculation
    try:
        auc = roc_auc_score(y_true, y_pred)
    except Exception:
        auc = 0  # Default for cases where AUC can't be calculated
    
    # Calculate confusion matrix elements
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Calculate G-mean
        g_mean = np.sqrt(sensitivity * specificity) if (sensitivity * specificity) > 0 else 0
    except ValueError:
        sensitivity = 0
        specificity = 0
        g_mean = 0
    
    return {
        'accuracy': acc,
        'auc': auc,
        'f1_score': f1,
        'g_mean': g_mean
    }

def count_selected_features(w):
    """
    Count the number of selected features
    
    Parameters:
    -----------
    w : numpy array
        Feature weights
        
    Returns:
    --------
    int
        Number of non-zero features
    list
        Indices of selected features (1-indexed)
    """
    if w is None:
        return 0, []
        
    selected_indices = np.where(np.abs(w) > 1e-3)[0]
    selected_features = [i + 1 for i in selected_indices]
    return len(selected_features), selected_features

def feature_selection_stability(feature_sets, n_features):
    """
    Calculate the stability of feature selection
    
    Parameters:
    -----------
    feature_sets : list of lists
        Each inner list contains the indices of selected features
    n_features : int
        Total number of features
        
    Returns:
    --------
    float
        Stability score between 0 and 1
    """
    if not feature_sets:
        return 0
    
    n_sets = len(feature_sets)
    if n_sets <= 1:
        return 1.0
    
    # Calculate pairwise Jaccard similarity
    jaccard_sum = 0
    count = 0
    
    for i in range(n_sets):
        for j in range(i + 1, n_sets):
            set_i = set(feature_sets[i])
            set_j = set(feature_sets[j])
            intersection = len(set_i.intersection(set_j))
            union = len(set_i.union(set_j))
            
            if union > 0:
                jaccard_sum += intersection / union
                count += 1
    
    if count == 0:
        return 0
        
    return jaccard_sum / count