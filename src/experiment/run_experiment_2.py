import os
import time
import numpy as np
from datetime import datetime
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils.data_loader import load_dataset, get_shape
from src.utils.preprocessing import standardize_data
from src.utils.metrics import evaluate_model, count_selected_features
from src.models.l1_svm import L1SVM
from src.models.l2_svm import L2SVM
from src.models.milp1_svm import MILP1
from src.models.pin_fs_svm import PinFSSVM
from src.models.pinball_svm import PinballSVM
from src.models.fisher_svm import FisherSVM
from src.models.rfe_svm import RFESVM
import matplotlib.pyplot as plt 
from joblib import Parallel, delayed


def save_auc_for_wilcoxon(model_class, dataset_name, dataset_type, best_params, 
                         best_cv_auc_scores, output_dir='results'):
    """
    Save AUC scores from each fold for Wilcoxon test analysis
    
    Parameters:
    -----------
    model_class : class
        The model class used
    dataset_name : str
        Name of the dataset
    dataset_type : str
        Type of dataset (original, noise, outlier, both)
    best_params : dict
        Best parameters found
    best_cv_auc_scores : list
        AUC scores from each fold for the best parameters
    output_dir : str
        Directory to save results
    """
    # Create wilcoxon directory
    wilcoxon_dir = os.path.join(output_dir, 'wilcoxon')
    os.makedirs(wilcoxon_dir, exist_ok=True)
    dataset_dir = os.path.join(wilcoxon_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    dirpath = os.path.join(dataset_dir, dataset_type)
    os.makedirs(dirpath, exist_ok=True)
    # Create filename with model, dataset, and dataset type
    timestamp = datetime.today().date()
    filename = f"{model_class.__name__}_auc_folds_{timestamp}.xlsx"
    filepath = os.path.join(dirpath, filename)
    
    # Prepare data for saving
    auc_data = {
        'Model': model_class.__name__,
        'Dataset': dataset_name,
        'Dataset_Type': dataset_type,
        'Fold': list(range(1, len(best_cv_auc_scores) + 1)),
        'AUC': best_cv_auc_scores
    }
    
    # Add best parameters as columns
    for param_name, param_value in best_params.items():
        auc_data[f'Best_{param_name}'] = [param_value] * len(best_cv_auc_scores)
    
    # Create DataFrame and save
    auc_df = pd.DataFrame(auc_data)
    auc_df.to_excel(filepath, index=False)
    
    print(f"  AUC fold data saved to: {filepath}")
    
    

def run_cv_for_params(model_class, params, X, y, kf):
    """
    Run cross-validation for a specific set of parameters
    
    Parameters:
    -----------
    model_class : class
        The model class to use
    params : dict
        Parameters for model initialization
    X, y : arrays
        Dataset and labels
    kf : StratifiedKFold
        Cross-validation splitter
    
    Returns:
    --------
    dict
        Results from cross-validation
    """
    n_splits = kf.get_n_splits()
    
    # Initialize metrics containers
    cv_metrics = {
        'accuracy': [], 'auc': [], 'f1_score': [], 'g_mean': [],
        'train_time': [], 'num_features': [],
        'optimization_gap': [], 'objective_value': [],
        'solver_status': []
    }
    cv_selected_features = []
    last_model = None
    best_fold_auc_single = -1
    best_fold_model_single = None
    best_fold_num_features_single = 0
    best_fold_selected_features_single = []
    
    # Perform cross-validation
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X,y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Standardize data
        X_train, X_test, _ = standardize_data(X_train, X_test)
        
        try:
            # Train model
            model = model_class(**params)
            model.fit(X_train, y_train)
            
            # Save last fold model
            if fold_idx == n_splits - 1:
                last_model = model
            
            # Predict and evaluate
            y_pred = model.predict(X_test)
            fold_metrics = evaluate_model(y_test, y_pred)
            
            # Store metrics
            for key in fold_metrics:
                cv_metrics[key].append(fold_metrics[key])
            
            # STORE GAP INFORMATION
            if hasattr(model, 'optimization_gap') and model.optimization_gap is not None:
                cv_metrics['optimization_gap'].append(model.optimization_gap)
            else:
                cv_metrics['optimization_gap'].append(0.0)  # Non-MILP models
                
            if hasattr(model, 'solver_status') and model.solver_status is not None:
                cv_metrics['solver_status'].append(model.solver_status)
            else:
                cv_metrics['solver_status'].append('N/A')
                
            if hasattr(model, 'objective_value') and model.objective_value is not None:
                cv_metrics['objective_value'].append(model.objective_value)
            else:
                cv_metrics['objective_value'].append(0.0)
            
            num_features_current_fold = 0
            selected_features_current_fold = []
            # Store feature selection info
            
            if model.__class__ == L2SVM or model.__class__ == PinballSVM:
                num_features_current_fold, selected_features_current_fold = model.w.shape[0], list(range(1, model.w.shape[0] + 1))
            else:
                if hasattr(model, 'v') and model.v is not None:
                    selected_features_current_fold = [j + 1 for j in range(len(model.v)) if model.v[j] > 0.5]
                    num_features_current_fold = len(selected_features_current_fold)
                else:
                    num_features_current_fold, selected_features_current_fold = count_selected_features(model.w)
            cv_metrics['train_time'].append(model.train_time)
            cv_metrics['num_features'].append(num_features_current_fold)
            cv_selected_features.append(selected_features_current_fold)
            
            current_fold_auc = fold_metrics['auc']
            if current_fold_auc > best_fold_auc_single:
                best_fold_auc_single = current_fold_auc
                best_fold_model_single = model
                best_fold_num_features_single = num_features_current_fold
                best_fold_selected_features_single = selected_features_current_fold
            
        except Exception as e:
            param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
            print(f"    Error with {param_str}, fold {fold_idx+1}: {e}")
            continue
    
    # Check if we have valid results
    if not cv_metrics['accuracy']:
        return None
    
    return {
        'metrics': cv_metrics, #metrics for all folds
        'selected_features_all_folds': cv_selected_features, #selected features for all folds
        'last_model': last_model, #model of the last fold
        'best_performing_fold_model': best_fold_model_single, #best performing fold model
        'best_performing_fold_num_features': best_fold_num_features_single, #best performing fold number of features
        'best_performing_fold_selected_features': best_fold_selected_features_single #best performing fold selected features
    }


def process_best_results(model_class, dataset_name, dataset_type, best_params, best_mean_cv_auc, 
                         best_mean_cv_accuracy, best_cv_f1_score, best_cv_g_mean, best_cv_metrics_all_folds, best_all_selected_features_all_folds, 
                         best_w, best_model, n_splits):
    """Process the best results from parameter grid search"""
    
    # Determine noise type
    noise_types = {
        'original': 'Not noise',
        'noise': 'Noise', 
        'outlier': 'Outlier',
        'both': 'Noise + Outlier'
    }
    noise_type = noise_types.get(dataset_type, 'Unknown')
    
    # Calculate feature frequency
    feature_freq = {}
    for features in best_all_selected_features_all_folds:
        for f in features:
            if f not in feature_freq:
                feature_freq[f] = 0
            feature_freq[f] += 1
    
    # Most frequently selected features
    frequent_features = sorted([(f, freq/n_splits) for f, freq in feature_freq.items()], 
                             key=lambda x: x[1], reverse=True)
        
    # Get final selected features
    if best_w is not None:
        if model_class == L2SVM or model_class == PinballSVM:
            selected_features_best_fold = [j + 1 for j in range(len(best_w))]
        else:
            n_features = len(best_w)
            selected_features_best_fold = [j + 1 for j in range(n_features) if abs(best_w[j]) > 1e-3]
    else:
        selected_features_best_fold = [f for f, freq in frequent_features if freq > 0.5]
    
    # Prepare result dictionary
    result = {
        'Model': model_class.__name__,
        'Type of dataset': noise_type,
        'Average Accuracy': best_mean_cv_accuracy,
        'Average AUC': best_mean_cv_auc,
        'Average F1 Score': best_cv_f1_score,
        'Average G-Mean': best_cv_g_mean,
    }
    
    # Add time and feature count
    for key in best_cv_metrics_all_folds:
        if key in ['train_time', 'num_features']:
            result[f'Average {key}'] = np.mean(best_cv_metrics_all_folds[key])
            # result[f'Std {key}'] = np.std(best_cv_metrics[key])
            
    
    # Add feature selection information
    result['BestFold Features selected'] = ', '.join(map(str, selected_features_best_fold))
    result['BestFold #Features'] = len(selected_features_best_fold)
    
    # Add best parameters
    for param_name, param_value in best_params.items():
        result[param_name] = param_value
        
    # Add gap information for MILP models  
    is_milp = model_class.__name__ in ['MILP1', 'PinFSSVM']
    
    if is_milp and 'optimization_gap' in best_cv_metrics_all_folds:
        gaps = [g for g in best_cv_metrics_all_folds['optimization_gap'] 
               if g != float('inf') and g >= 0]
        
        if gaps:
            avg_gap = np.mean(gaps)
            max_gap = np.max(gaps)
            std_gap = np.std(gaps)
            
            result['Average optimization_gap'] = f"{avg_gap*100:.3f}%"
            result['Max optimization_gap'] = f"{max_gap*100:.3f}%"
            result['Std optimization_gap'] = f"{std_gap*100:.3f}%"
            
        else:
            result['Average optimization_gap'] = 'N/A'
            result['Max optimization_gap'] = 'N/A'
            result['Std optimization_gap'] = 'N/A'
    # Print results
    param_str = ", ".join([f"{k}={v}" for k, v in best_params.items()])
    print(f"  Best parameters: {param_str}")
    print(f"  Accuracy={best_mean_cv_accuracy:.4f}, AUC={best_mean_cv_auc:.4f}, F1={best_cv_f1_score:.4f}, G-Mean={best_cv_g_mean:.4f}")
    print(f"  Average #Features: {np.mean(best_cv_metrics_all_folds['num_features']):.2f}")
    print(f"  BestFold Features selected: {selected_features_best_fold}")
    print(f"  BestFold Number of selected features: {len(selected_features_best_fold)}")
    print('-' * 80)
    
    return result, frequent_features, selected_features_best_fold


def save_detailed_metrics(model_class, dataset_name, dataset_type, all_param_metrics, 
                          output_dir='results'):
    """Save detailed metrics for all parameter combinations"""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    param_metrics_dir = os.path.join(output_dir, 'detailed_metrics')
    os.makedirs(param_metrics_dir, exist_ok=True)
    # if dataset_name == 'colon':
    #     n_splits = 5
    # else:
    n_splits = 10
    for param_key, metrics_data in all_param_metrics.items():
        detailed_file = os.path.join(
            param_metrics_dir, 
            f"{model_class.__name__}_{dataset_name}_{dataset_type}_{param_key}_{timestamp}.xlsx"
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
        
        # Add selected features
        for fold_idx, features in enumerate(metrics_data['selected_features']):
            metrics_df.at[fold_idx, 'selected_features'] = str(features)
        
        metrics_df.to_excel(detailed_file, index=False)



def run_grid_search(model_class, param_values, dataset_name, dataset_type, 
                   X, y, kf, output_dir='results', fixed_params=None, overall_time_limit=None):
    """
    Run grid search for parameter optimization
    
    Parameters:
    -----------
    model_class : class
        Model class to optimize
    param_values : dict
        Parameter grid with parameter names as keys and lists of values
    dataset_name, dataset_type : str
        Dataset information
    X, y : arrays
        Dataset features and labels
    kf : StratifiedKFold
        Cross-validation splitter
    output_dir : str
        Directory to save results
    fixed_params : dict or None
        Fixed parameters to use with all parameter combinations
    
    Returns:
    --------
    dict
        Best results
    """
    n_splits = kf.get_n_splits()
    param_names = list(param_values.keys())
    
    # Initialize tracking
    best_params = {}
    best_mean_cv_auc = 0
    best_mean_cv_accuracy = 0
    best_mean_cv_f1_score = 0
    best_mean_cv_g_mean = 0
    best_cv_metrics = None
    best_all_selected_features = []
    best_w = None
    best_model = None
    best_model_overall_best_fold = None
    best_cv_auc_scores = []
    all_param_metrics = {}
    all_param_metrics = {}
    # Generate all parameter combinations
    import itertools
    param_combinations = []
    param_values_list = [param_values[param] for param in param_names]
    
    for values in itertools.product(*param_values_list):
        current_params = {param_names[i]: values[i] for i in range(len(param_names))}
        if fixed_params:
            current_params.update(fixed_params)
        param_combinations.append(current_params)
    num_parallel_jobs = 20
    batch_size = num_parallel_jobs
    param_batches = [param_combinations[i:i + batch_size] for i in range(0, len(param_combinations), batch_size)]
    grid_search_start_time = time.time()
    all_completed_results = []
    print(f"Starting grid search for {model_class.__name__} on {dataset_name} ({dataset_type}) with {len(param_combinations)} combinations")
    def run_cv_for_single_param(params_evaluated_in_job):
        cv_results_from_func = run_cv_for_params(
            model_class, params_evaluated_in_job, X, y, kf
        )
        return params_evaluated_in_job, cv_results_from_func
    # parallel_execution_results = Parallel(n_jobs=num_parallel_jobs, verbose=10, backend='loky')(
    #     delayed(run_cv_for_single_param)(p_combo) for p_combo in param_combinations
    # )
    for i, p_batch in enumerate(param_batches):
        elapsed_in_grid_search = time.time() - grid_search_start_time
        if overall_time_limit is not None and elapsed_in_grid_search >= overall_time_limit:
            print(f"  Total time limit reached for grid search. Stopping further batches. Processed {i}/{len(param_batches)} batches.")
            break
        print(f"    Running batch {i+1}/{len(param_batches)}...")
        
        batch_results = Parallel(n_jobs=num_parallel_jobs, verbose=0, backend='loky')( # giảm verbose để không quá nhiều output
            delayed(run_cv_for_single_param)(p_combo) for p_combo in p_batch
        )
        all_completed_results.extend(batch_results)
    if not all_completed_results:
        print("  No parameter combinations were evaluated before timeout.")
        return None
    print(f"\n  Grid search finished. Evaluating {len(all_completed_results)} completed combinations.")
    # Evaluate each parameter combination
    for params, cv_results in all_completed_results:
        # Create a parameter key for saving
        
        if cv_results:
            cv_metrics = cv_results['metrics']
            mean_cv_accuracy = np.mean(cv_metrics['accuracy'])
            mean_cv_auc = np.mean(cv_metrics['auc'])
            mean_f1_score = np.mean(cv_metrics['f1_score'])
            mean_g_mean = np.mean(cv_metrics['g_mean'])
            
            # CALCULATE GAP STATISTICS
            gaps = [g for g in cv_metrics.get('optimization_gap', []) if g != float('inf') and g >= 0]
            avg_gap_percent = np.mean(gaps) * 100 if gaps else 0
            max_gap_percent = np.max(gaps) * 100 if gaps else 0
            
            # DETERMINE IF MILP MODEL
            is_milp = model_class.__name__ in ['MILP1', 'PinFSSVM']
            
            # Print results
            param_key = "_".join([f"{k}{v}" for k, v in params.items()])
            param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
            if is_milp and gaps:
                if avg_gap_percent > 0.01:
                    print(f"  {param_str}: Acc={mean_cv_accuracy:.4f}, AUC={mean_cv_auc:.4f}, F1={mean_f1_score:.4f}, G-Mean={mean_g_mean:.4f}, Avg Gap={avg_gap_percent:.2f}%")
                else:
                    print(f"  {param_str}: Acc={mean_cv_accuracy:.4f}, AUC={mean_cv_auc:.4f}, F1={mean_f1_score:.4f}, G-Mean={mean_g_mean:.4f}, Avg Gap=Exact")
            else:
                print(f"  {param_str}: Acc={mean_cv_accuracy:.4f}, AUC={mean_cv_auc:.4f}, F1={mean_f1_score:.4f}, G-Mean={mean_g_mean:.4f}")
            
            # Store metrics for this parameter set
            all_param_metrics[param_key] = {
                'metrics': cv_metrics,
                'selected_features': cv_results['selected_features_all_folds']
            }
            
            # Update best if better
            if mean_cv_auc > best_mean_cv_auc:
                best_params = params.copy()
                best_mean_cv_auc = mean_cv_auc
                best_mean_cv_accuracy = mean_cv_accuracy
                best_mean_cv_f1_score = mean_f1_score
                best_mean_cv_g_mean = mean_g_mean
                best_cv_metrics = cv_metrics
                best_all_selected_features = cv_results['selected_features_all_folds']
                best_cv_auc_scores = cv_metrics['auc']
                
                best_model_overall_best_fold = cv_results['best_performing_fold_model']
                num_features_overall_best_fold = cv_results['best_performing_fold_num_features']
                best_selected_features_overall_best_fold = cv_results['best_performing_fold_selected_features']
    
    # Process best results
    if best_params:
        best_w = None
        if best_model_overall_best_fold:
            best_w = best_model_overall_best_fold.w
            best_model = best_model_overall_best_fold
            
        result, frequent_features, final_selected_features = process_best_results(
            model_class, dataset_name, dataset_type, best_params, 
            best_mean_cv_auc, best_mean_cv_accuracy, best_mean_cv_f1_score, best_mean_cv_g_mean, best_cv_metrics,
            best_all_selected_features, best_w, best_model, n_splits
        )
        
        # Save detailed metrics
        # save_detailed_metrics(
        #     model_class, dataset_name, dataset_type, 
        #     all_param_metrics, output_dir
        # )
        # Save AUC scores for Wilcoxon test
        save_auc_for_wilcoxon(
            model_class, dataset_name, dataset_type, 
            best_params, best_cv_auc_scores, output_dir
        )
        
        return {
            'result': result,
            'frequent_features': frequent_features,
            'final_selected_features': final_selected_features,
            'all_selected_features': best_all_selected_features,
            'best_w': best_w,
            'best_model': best_model,
            'best_auc_scores': best_cv_auc_scores
            
        }
    
    return None


def run_experiment(models_config, datasets_config, output_dir='results', time_limit = 5 * 3600 ):
    """
    Run experiments with multiple models and datasets
    
    Parameters:
    -----------
    models_config : list of dict
        Each dict contains model_class and param_grid
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
    for dataset_config in datasets_config:
        dataset_name = dataset_config['dataset_name']
        dataset_types = dataset_config.get('dataset_types', ['original'])
        
        print(f"\nRunning experiments for dataset: {dataset_name}")
        print(f"Dataset types: {dataset_types}")
        print(f"Time limit per dataset type this: {time_limit / 3600:.2f} hours")
        print('-' * 80)
        for dataset_type in dataset_types:
            print(f"  Dataset type: {dataset_type}")
            start_time = time.time()
            for model_config in models_config:
                model_class = model_config['model_class']
                param_grid = model_config.get('param_grid', {})
                fixed_params = model_config.get('fixed_params', {})
            
                print(f"\nRunning {model_class.__name__} on {dataset_name} ({dataset_type})")
                
                # Load dataset
                X, y = load_dataset(dataset_name, dataset_type)
                if X.size == 0 or y.size == 0:
                    print(f"Failed to load dataset {dataset_name} ({dataset_type})")
                    continue
                n_splits = 5 if dataset_name == 'colon' else 10
                kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                
                # Run grid search
                result_data = run_grid_search(
                    model_class, param_grid, dataset_name, dataset_type,
                    X, y, kf, output_dir, fixed_params,
                    overall_time_limit= time_limit
                )
                
                if result_data:
                    all_results.append(result_data['result'])
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"  Completed dataset type {dataset_type} for {dataset_name} in {elapsed_time / 60:.2f} minutes")
            print('-' * 80)
    
    # Check if we have results
    if not all_results:
        print("No results were generated.")
        return None
    
    # Convert to DataFrame and save
    results_df = pd.DataFrame(all_results)
    timestamp = datetime.today().date()
    results_path = os.path.join(output_dir, f'experiment_results_{dataset_name}_{timestamp}_pinfssvm_5h.xlsx')
    results_df.to_excel(results_path, index=False)
    
    print(f"\nSummary results saved to {results_path}")
    # print(f"Detailed metrics saved in {os.path.join(output_dir, 'detailed_metrics')}")
    
    return results_df


if __name__ == '__main__':
# Datasets to test
    data_config = [
        {
            'dataset_name': 'colon',
            'dataset_types': ['original', 'noise', 'outlier', 'both']
        }
    ]
    m, n = get_shape(data_config[0]['dataset_name'])
    print(f"Dataset shape: {m} samples, {n} features")
    # Define models with parameter grids to test
    models_config = [
        # {
        #     'model_class': L1SVM,
        #     'param_grid': {
        #         'C': [2**i for i in range(-3, 6)]  # C from 2^-3 to 2^5
        #     }
        # },
        # {
        #     'model_class': L2SVM,
        #     'param_grid': {
        #         'C': [2**i for i in range(-3, 6)]  # C from 2^-3 to 2^5
        #     }
        # },
        # {
        #     'model_class': MILP1,
        #     'param_grid': {
        #         'C': [2**i for i in range(-3, 6)],  # C from 2^-3 to 2^5
        #         'B': [i for i in range(1, n+1)],     # B is max number of features
        #         'cpu_threads': [1], 
        #         'time_limit': [60] 
        #     }
        # },
        {
            'model_class': PinFSSVM,
            'param_grid': {
                'C': [2**i for i in range(-3, 6)],  # C from 2^-3 to 2^5
                'tau': [0.1, 0.5, 1.0],            # Pinball loss parameter
                'B': [i for i in range(1, n+1)],       # B is max number of features
                'cpu_threads': [1],
                'time_limit': [60] 
            },
    
        },
        # {
        #     'model_class': PinballSVM,
        #     'param_grid': {
        #         'C': [2**i for i in range(-3, 6)],  # C from 2^-3 to 2^5
        #         'tau': [0.1, 0.5, 1.0],           
        #         'cpu_threads': [1], 
        #     }
        # },
        # {
        #      'model_class': FisherSVM,
        #     'param_grid': {
        #         'C': [2**i for i in range(-3, 6)],  # C from 2^-3 to 2^5,
        #         'time_limit': [60], 
               
        #     }
        # },
        # {
        #     'model_class': RFESVM,
        #     'param_grid': {
        #         'C': [2**i for i in range(-3, 6)],  # C from 2^-3 to 2^5
        #         'n_features': [n//4,n//2,int(3*n/4)],  # Number of features to select
        #         'time_limit': [60],
        #     }
        # }
    ]

    
    # Run experiments
    results = run_experiment(models_config, data_config, output_dir=f'results', time_limit=5 * 3600)