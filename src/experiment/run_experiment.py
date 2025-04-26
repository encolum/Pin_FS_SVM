import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
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
    Run a single experiment with a specific model and dataset
    
    Parameters:
    -----------
    model_class : class
        The model class to use
    model_params : dict
        Parameters for the model initialization
    dataset_name : str
        Name of the dataset to use
    dataset_type : str, default='original'
        Type of dataset (original, noise, outlier, both)
    n_splits : int, default=10
        Number of cross-validation splits
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns:
    --------
    dict
        Results dictionary with performance metrics
    """
    # Load dataset
    X, y = load_dataset(dataset_name, dataset_type)
    if X.size == 0 or y.size == 0:
        print(f"Failed to load dataset {dataset_name} ({dataset_type})")
        return None
        
    print(f"Running {model_class.__name__} on {dataset_name} ({dataset_type})")
    
    # Setup cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Initialize metrics containers
    metrics = {
        'accuracy': [],
        'auc': [],
        'f1_score': [],
        'g_mean': [],
        'train_time': [],
        'num_features': []
    }
    all_selected_features = []
    
    # Perform cross-validation
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"  Fold {fold_idx+1}/{n_splits}")
        
        # Split and standardize data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        X_train, X_test, _ = standardize_data(X_train, X_test)
        
        try:
            # Train model
            model = model_class(**model_params)
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
    
            
            # Evaluate
            fold_metrics = evaluate_model(y_test, y_pred)
            for key in fold_metrics:
                metrics[key].append(fold_metrics[key])
            
            # Save time and selected features
            metrics['train_time'].append(model.train_time)
            num_features, selected_features = count_selected_features(model.w)
            metrics['num_features'].append(num_features)
            all_selected_features.append(selected_features)
            
        except Exception as e:
            print(f"    Error in fold {fold_idx+1}: {e}")
    
    # Check if we have valid results
    if not metrics['accuracy']:
        print("No valid results obtained.")
        return None
    
    # Prepare result dictionary
    result = {
        'model': model_class.__name__,
        'dataset': dataset_name,
        'dataset_type': dataset_type,
        'params': model_params,
    }
    
    # Add metrics
    for key in metrics:
        result[f'{key}_mean'] = np.mean(metrics[key])
        if key in ['accuracy', 'auc']:  # Only add std for main metrics
            result[f'{key}_std'] = np.std(metrics[key])
    
    # Add feature selection information
    result['selected_features'] = all_selected_features
    result['stability'] = feature_selection_stability(all_selected_features, X.shape[1])
    
    return result

    
def run_full_experiment(models_config, datasets_config, output_dir='results'):
    """
    Run a full experiment with multiple models and datasets, saving detailed metrics
    
    Parameters:
    -----------
    models_config : list of dict
        Each dict contains model_class, model_params_grid
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
        
        # Check if parameter optimization is needed
        if 'param_grid' in model_config:
            param_name = list(model_config['param_grid'].keys())[0]
            param_values = model_config['param_grid'][param_name]
            
            for dataset_config in datasets_config:
                dataset_name = dataset_config['dataset_name']
                dataset_types = dataset_config.get('dataset_types', ['original'])
                
                for dataset_type in dataset_types:
                    print(f"Finding best {param_name} for {model_class.__name__} on {dataset_name} ({dataset_type})")
                    
                    # Load dataset
                    X, y = load_dataset(dataset_name, dataset_type)
                    
                    if X.size == 0 or y.size == 0:
                        print(f"Failed to load dataset {dataset_name} ({dataset_type})")
                        continue
                    
                    # Setup cross-validation
                    n_splits = 10
                    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                    
                    best_param_value = None
                    best_mean_cv_auc = 0
                    best_mean_cv_accuracy = 0
                    best_cv_metrics = None
                    best_all_selected_features = []
                    best_w = None
                    best_model = None
                    
                    # Track detailed metrics for all parameter values
                    all_param_metrics = {}
                    
                    for param_value in param_values:
                        # Create parameter dictionary
                        current_params = {param_name: param_value}
                        
                        # Add fixed parameters if available
                        if 'fixed_params' in model_config:
                            current_params.update(model_config['fixed_params'])
                        
                        # Evaluate current parameter with cross-validation
                        cv_metrics = {
                            'accuracy': [],
                            'auc': [],
                            'f1_score': [],
                            'g_mean': [],
                            'train_time': [],
                            'num_features': []
                        }
                        cv_selected_features = []
                        all_w_values = []
                        
                        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
                            X_train, X_test = X[train_idx], X[test_idx]
                            y_train, y_test = y[train_idx], y[test_idx]
                            
                            # Standardize data
                            X_train, X_test, _ = standardize_data(X_train, X_test)
                            
                            try:
                                # Train model with current parameter
                                model = model_class(**current_params)
                                model.fit(X_train, y_train)
                                
                                # Save model weights for the last fold
                                if fold_idx == n_splits-1:
                                    last_fold_model = model
                                
                                # Predict
                                y_pred = model.predict(X_test)
                                
                                # Evaluate
                                fold_metrics = evaluate_model(y_test, y_pred)
                                for key in fold_metrics:
                                    cv_metrics[key].append(fold_metrics[key])
                                
                                # Save time and selected features
                                cv_metrics['train_time'].append(model.train_time)
                                num_features, selected_features = count_selected_features(model.w)
                                cv_metrics['num_features'].append(num_features)
                                cv_selected_features.append(selected_features)
                                all_w_values.append(model.w)
                                
                            except Exception as e:
                                print(f"    Error with {param_name}={param_value}, fold {fold_idx+1}: {e}")
                        
                        # Calculate average metrics across folds
                        if cv_metrics['accuracy']:
                            mean_cv_accuracy = np.mean(cv_metrics['accuracy'])
                            mean_cv_auc = np.mean(cv_metrics['auc'])
                            print(f"  {param_name}={param_value}: Accuracy={mean_cv_accuracy:.4f}, AUC={mean_cv_auc:.4f}")
                            
                            # Store detailed metrics for this parameter
                            all_param_metrics[param_value] = {
                                'metrics': cv_metrics,
                                'selected_features': cv_selected_features,
                                'w_values': all_w_values
                            }
                            
                            # Update best parameter if AUC is higher
                            if mean_cv_auc > best_mean_cv_auc:
                                best_param_value = param_value
                                best_mean_cv_auc = mean_cv_auc
                                best_mean_cv_accuracy = mean_cv_accuracy
                                best_cv_metrics = cv_metrics.copy()
                                best_all_selected_features = cv_selected_features
                                best_w = last_fold_model.w if 'last_fold_model' in locals() else None
                                best_model = last_fold_model if 'last_fold_model' in locals() else None
                    
                    if best_param_value is not None:
                        
                        # Calculate feature selection stability
                        stability = feature_selection_stability(best_all_selected_features, X.shape[1])
                        
                        # Calculate feature frequency
                        feature_freq = {}
                        for features in best_all_selected_features:
                            for f in features:
                                if f not in feature_freq:
                                    feature_freq[f] = 0
                                feature_freq[f] += 1
                        
                        # Most frequently selected features
                        frequent_features = sorted([(f, freq/n_splits) for f, freq in feature_freq.items()], 
                                                 key=lambda x: x[1], reverse=True)
                        
                        # Get final selected features if best_w is available
                        if best_w is not None:
                            n_features = len(best_w)
                            final_selected_features = [j + 1 for j in range(n_features) if abs(best_w[j]) > 1e-6]
                        else:
                            final_selected_features = []
                            for f, freq in frequent_features:
                                if freq > 0.5:  # Features selected in more than 50% of folds
                                    final_selected_features.append(f)
                        
                        # Prepare result dictionary
                        result = {
                            'model': model_class.__name__,
                            'dataset': dataset_name,
                            'dataset_type': dataset_type,
                            'params': {param_name: best_param_value},
                            'stability': stability,
                            'accuracy': best_mean_cv_accuracy,
                            'auc': best_mean_cv_auc
                        }
                        
                        # Add metrics from best parameter
                        for key in best_cv_metrics:
                            result[f'{key}'] = np.mean(best_cv_metrics[key])
                            # result[f'{key}_std'] = np.std(best_cv_metrics[key])
                        
                        # Add feature selection information
                        result['frequent_features'] = frequent_features
                        result['all_selected_features'] = best_all_selected_features
                        result['final_selected_features'] = final_selected_features
                        result['number_of_features'] = len(final_selected_features)
                        result['features_selected'] = ', '.join(map(str, final_selected_features))
                        
                        all_results.append(result)
                        print(f"  Best {param_name}={best_param_value}, Accuracy={result['accuracy']:.4f}, AUC={result['auc']:.4f}")
                        print(f"  Features selected: {final_selected_features}")
                        print(f"  Number of selected features: {len(final_selected_features)}")
                        
                        # Save detailed metrics for all parameter values
                        timestamp = time.strftime("%Y%m%d-%H%M%S")
                        param_metrics_dir = os.path.join(output_dir, 'detailed_metrics')
                        os.makedirs(param_metrics_dir, exist_ok=True)
                        
                        for param_value, metrics_data in all_param_metrics.items():
                            detailed_file = os.path.join(
                                param_metrics_dir, 
                                f"{model_class.__name__}_{dataset_name}_{dataset_type}_{param_name}_{param_value}_{timestamp}.csv"
                            )
                            
                            # Create detailed metrics DataFrame
                            metrics_df = pd.DataFrame({
                                'fold': list(range(1, n_splits+1)),
                                'accuracy': metrics_data['metrics']['accuracy'],
                                'auc': metrics_data['metrics']['auc'],
                                'f1_score': metrics_data['metrics']['f1_score'],
                                'g_mean': metrics_data['metrics']['g_mean'],
                                'train_time': metrics_data['metrics']['train_time'],
                                'num_features': metrics_data['metrics']['num_features']
                            })
                            
                            # Add selected features as columns
                            for fold_idx, features in enumerate(metrics_data['selected_features']):
                                metrics_df.at[fold_idx, 'selected_features'] = str(features)
                            
                            metrics_df.to_csv(detailed_file, index=False)
    
    # Check if we have results
    if not all_results:
        print("No results were generated. Please check your dataset paths and model configurations.")
        return None
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Clean DataFrame for saving
    save_cols = [col for col in results_df.columns if not col.endswith('_per_fold') 
                and col not in ['all_selected_features', 'frequent_features']]
    
    save_df = results_df[save_cols].copy()
    
    # Save summary results
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_path = os.path.join(output_dir, f'experiment_results_{timestamp}.csv')
    save_df.to_csv(results_path, index=False)
    
    print(f"Summary results saved to {results_path}")
    print(f"Detailed per-fold metrics saved in {os.path.join(output_dir, 'detailed_metrics')}")
    
    return results_df
if __name__ == '__main__':
    # Ví dụ cách sử dụng để tìm tham số C tốt nhất cho L1SVM
    models_config = [
        # {
        #     'model_class': L1SVM,
        #     'param_grid': {
        #         'C': [2**i for i in range(-3, 6)]  # C từ 2^-3 đến 2^5
        #     }
        # },
        
        {
            
            'model_class': L2SVM,
            'param_grid': {
                'C': [2**i for i in range(-3, 6)]  # C từ 2^-3 đến 2^5
            }
        },
        # {
        #     'model_class': MILP1,
        #     'param_grid': {
        #         'C': [2**i for i in range(-3, 6)],  # C từ 2^-3 đến 2^5,
        #         'B': [i for i in range(1,9)],  # B là số lượng đặc trưng tối đa)
        #     }
        # },
        # {
        #     'model_class': PinFSSVM,
        #     'param_grid': {
        #         'C': [2**i for i in range(-3, 6)],  # C từ 2^-3 đến 2^5
        #         'tau': [0.1, 0.5, 1.0]  # Thay đổi giá trị tau
        #     }
        # },
        # {
        #     'model_class': PinballSVM,
        #     'param_grid': {
        #         'C': [2**i for i in range(-3, 6)],  # C từ 2^-3 đến 2^5
        #         'tau': [0.1, 0.5, 1.0]  # Thay đổi giá trị tau
        #     }
        # },
        # {
        #     'model_class': FisherSVM,
        #     'param_grid': {
        #         'C': [2**i for i in range(-3, 6)]  # C từ 2^-3 đến 2^5
        #     }
        # },
        # {
        #     'model_class': RFESVM,
        #     'param_grid': {
        #         'C': [2**i for i in range(-3, 6)]  # C từ 2^-3 đến 2^5
        #     }
        # }
    ]
    
    
    data_config = [
        {
            'dataset_name': 'diabetes',
            'dataset_types': ['original', 'noise', 'outlier', 'both']
        }
    ]
    
    results = run_full_experiment(models_config, data_config, output_dir='results')