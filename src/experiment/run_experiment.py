import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from src.utils.data_loader import load_dataset
from src.utils.preprocessing import standardize_data
from src.utils.metrics import evaluate_model, count_selected_features, feature_selection_stability
from src.models.l1_svm import L1SVM
from src.models.l2_svm import L2SVM
from src.models.milp1_svm import MILP1
from src.models.pin_fs_svm import PinFSSVM
from src.models.pinball_svm import PinballSVM
from src.models.fisher_svm import FisherSVM
from src.models.rfe_svm import RFESVM

def run_single_experiment(model_class, model_params, dataset_name, dataset_type="original", n_splits=10, random_state=42):
    """
    Run a single experiment with cross-validation
    
    Parameters:
    -----------
    model_class : class
        The model class to use
    model_params : dict
        Model parameters
    dataset_name : str
        Name of the dataset
    dataset_type : str
        Type of dataset ('original', 'noise', 'outlier', 'both')
    n_splits : int
        Number of cross-validation folds
    random_state : int
        Random seed
    
    Returns:
    --------
    dict
        Results dictionary
    """
    # Load dataset
    X, y = load_dataset(dataset_name, dataset_type)
    
    if X.size == 0 or y.size == 0:
        print(f"Failed to load dataset {dataset_name} ({dataset_type})")
        return None
    
    print(f"Loaded dataset {dataset_name} ({dataset_type}) with {X.shape[0]} samples and {X.shape[1]} features")
    
    # Setup cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Results collection
    cv_metrics = {
        'accuracy': [],
        'balanced_accuracy': [],
        'auc': [],
        'sensitivity': [],
        'specificity': [],
        'train_time': [],
        'num_features': []
    }
    all_selected_features = []
    
    # Cross-validation loop
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"  Fold {fold_idx+1}/{n_splits}")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Standardize data
        X_train, X_test, _ = standardize_data(X_train, X_test)
        
        # Instantiate and fit model
        model = model_class(**model_params)
        try:
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_score = model.decision_function(X_test)
            
            # Evaluate
            metrics = evaluate_model(y_test, y_pred, y_score)
            for key in metrics:
                cv_metrics[key].append(metrics[key])
            
            # Store training time and selected features
            cv_metrics['train_time'].append(model.train_time)
            num_features, selected_features = count_selected_features(model.w)
            cv_metrics['num_features'].append(num_features)
            all_selected_features.append(selected_features)
            
        except Exception as e:
            print(f"    Error in fold {fold_idx+1}: {e}")
    
    # Check if we have any results
    if not cv_metrics['accuracy']:
        print(f"No valid results for {model_class.__name__} on {dataset_name} ({dataset_type})")
        return None
    
    # Calculate feature selection stability
    stability = feature_selection_stability(all_selected_features, X.shape[1])
    
    # Calculate feature frequency
    feature_freq = {}
    for features in all_selected_features:
        for f in features:
            if f not in feature_freq:
                feature_freq[f] = 0
            feature_freq[f] += 1
    
    # Most frequently selected features
    frequent_features = sorted([(f, freq/n_splits) for f, freq in feature_freq.items()], 
                              key=lambda x: x[1], reverse=True)
    
    # Aggregate results
    results = {
        'model': model_class.__name__,
        'dataset': dataset_name,
        'dataset_type': dataset_type,
        'params': model_params,
        'stability': stability
    }
    
    # Add mean and std for each metric
    for key in cv_metrics:
        if cv_metrics[key]:
            results[f'{key}_mean'] = np.mean(cv_metrics[key])
            results[f'{key}_std'] = np.std(cv_metrics[key])
        else:
            results[f'{key}_mean'] = np.nan
            results[f'{key}_std'] = np.nan
    
    # Add feature information
    results['frequent_features'] = frequent_features
    results['all_selected_features'] = all_selected_features
    
    return results

def run_full_experiment(models_config, datasets_config, output_dir='results'):
    """
    Run a full experiment with multiple models and datasets
    
    Parameters:
    -----------
    models_config : list of dict
        Each dict contains model_class and model_params
    datasets_config : list of dict
        Each dict contains dataset_name and dataset_types
    output_dir : str
        Directory to save results
    
    Returns:
    --------
    pd.DataFrame
        Results dataframe
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    
    for model_config in models_config:
        model_class = model_config['model_class']
        model_params_list = model_config.get('model_params_list', [{}])
        
        for model_params in model_params_list:
            for dataset_config in datasets_config:
                dataset_name = dataset_config['dataset_name']
                dataset_types = dataset_config.get('dataset_types', ['original'])
                
                for dataset_type in dataset_types:
                    print(f"Running {model_class.__name__} on {dataset_name} ({dataset_type})")
                    try:
                        result = run_single_experiment(
                            model_class=model_class,
                            model_params=model_params,
                            dataset_name=dataset_name,
                            dataset_type=dataset_type
                        )
                        
                        if result:
                            all_results.append(result)
                            print(f"  Completed: Accuracy={result['accuracy_mean']:.4f}, AUC={result['auc_mean']:.4f}, Features={result['num_features_mean']:.1f}")
                        else:
                            print(f"  Failed to complete experiment")
                    except Exception as e:
                        print(f"  Error: {e}")
    
    # Check if we have any results
    if not all_results:
        print("No results were generated. Please check your dataset paths and model configurations.")
        return None
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Clean up the dataframe for saving
    save_cols = [col for col in results_df.columns if col not in ['all_selected_features', 'frequent_features', 'params']]
    save_df = results_df[save_cols].copy()
    
    # Save results
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_path = os.path.join(output_dir, f'experiment_results_{timestamp}.csv')
    save_df.to_csv(results_path, index=False)
    
    print(f"Results saved to {results_path}")
    return results_df

if __name__ == "__main__":
    # Example usage
    models_config = [
        {
            'model_class': PinFSSVM,
            'model_params_list': [
                {'B': 5, 'C': 1.0, 'tau': 0.1},
                {'B': 5, 'C': 1.0, 'tau': 0.5},
                {'B': 5, 'C': 1.0, 'tau': 1.0}
            ]
        },
        {
            'model_class': MILP1,
            'model_params_list': [
                {'B': 5, 'C': 1.0}
            ]
        },
        {
            'model_class': L1SVM,
            'model_params_list': [
                {'C': 1.0}
            ]
        },
        {
            'model_class': L2SVM,
            'model_params_list': [
                {'C': 1.0}
            ]
        },
        {
            'model_class': FisherSVM,
            'model_params_list': [
                {'n_features': 5, 'C': 1.0}
            ]
        },
        {
            'model_class': RFESVM,
            'model_params_list': [
                {'n_features': 5, 'C': 1.0}
            ]
        }
    ]
    
    datasets_config = [
        {
            'dataset_name': 'wdbc',
            'dataset_types': ['original', 'noise', 'outlier', 'both']
        },
        {
            'dataset_name': 'diabetes',
            'dataset_types': ['original', 'noise', 'outlier', 'both']
        }
    ]
    
    results = run_full_experiment(models_config, datasets_config)
