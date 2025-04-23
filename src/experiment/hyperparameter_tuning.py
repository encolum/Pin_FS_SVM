import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from src.utils.data_loader import load_dataset
from src.utils.preprocessing import standardize_data
from src.utils.metrics import evaluate_model
from src.models.pin_fs_svm import PinFSSVM

def tune_C_parameter(dataset_name, dataset_type="original", B=None, tau=0.5, 
                   C_values=None, n_splits=5, random_state=42):
    """
    Tune the C parameter for Pin-FS-SVM
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset
    dataset_type : str
        Type of dataset ('original', 'noise', 'outlier', 'both')
    B : int or None
        Number of features to select
    tau : float
        Pinball loss parameter
    C_values : list or None
        List of C values to try. If None, uses default range
    n_splits : int
        Number of cross-validation folds
    random_state : int
        Random seed
    
    Returns:
    --------
    pd.DataFrame
        Results dataframe
    dict
        Best parameters
    """
    # Default C values if not provided
    if C_values is None:
        C_values = [0.01, 0.1, 1, 10, 100]
    
    # Load dataset
    X, y = load_dataset(dataset_name, dataset_type)
    
    if X.size == 0 or y.size == 0:
        print(f"Failed to load dataset {dataset_name} ({dataset_type})")
        return None, None
        
    print(f"Tuning C parameter for {dataset_name} ({dataset_type}) dataset")
    print(f"Dataset shape: {X.shape}, Testing C values: {C_values}")
    
    # Setup cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Results storage
    results = []
    
    for C in C_values:
        print(f"  Testing C={C}")
        cv_metrics = {
            'accuracy': [],
            'auc': [],
            'train_time': [],
            'num_features': []
        }
        
        # Cross-validation
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            print(f"    Fold {fold_idx+1}/{n_splits}")
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Standardize data
            X_train, X_test, _ = standardize_data(X_train, X_test)
            
            try:
                # Train model
                model = PinFSSVM(B=B, C=C, tau=tau)
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_test)
                y_score = model.decision_function(X_test)
                
                # Evaluate
                metrics = evaluate_model(y_test, y_pred, y_score)
                cv_metrics['accuracy'].append(metrics['accuracy'])
                cv_metrics['auc'].append(metrics['auc'])
                cv_metrics['train_time'].append(model.train_time)
                cv_metrics['num_features'].append(model.get_num_selected_features())
            except Exception as e:
                print(f"    Error in fold {fold_idx+1}: {e}")
        
        # Skip if no results for this C value
        if not cv_metrics['accuracy']:
            print(f"  No valid results for C={C}")
            continue
            
        # Aggregate results for this C value
        result = {
            'C': C,
            'accuracy_mean': np.mean(cv_metrics['accuracy']),
            'accuracy_std': np.std(cv_metrics['accuracy']),
            'auc_mean': np.mean(cv_metrics['auc']),
            'auc_std': np.std(cv_metrics['auc']),
            'train_time_mean': np.mean(cv_metrics['train_time']),
            'num_features_mean': np.mean(cv_metrics['num_features'])
        }
        results.append(result)
        print(f"  C={C}: Accuracy={result['accuracy_mean']:.4f}, AUC={result['auc_mean']:.4f}")
    
    # Check if we have any results
    if not results:
        print("No valid results were obtained during parameter tuning")
        return None, None
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Find best C value
    best_idx = results_df['auc_mean'].idxmax()
    best_params = {
        'C': results_df.iloc[best_idx]['C'],
        'auc_mean': results_df.iloc[best_idx]['auc_mean']
    }
    
    print(f"Best C value: {best_params['C']} with AUC: {best_params['auc_mean']:.4f}")
    
    return results_df, best_params

def tune_tau_parameter(dataset_name, dataset_type="original", B=None, C=1.0, 
                    tau_values=None, n_splits=5, random_state=42):
    """
    Tune the tau parameter for Pin-FS-SVM
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset
    dataset_type : str
        Type of dataset ('original', 'noise', 'outlier', 'both')
    B : int or None
        Number of features to select
    C : float
        Regularization parameter
    tau_values : list or None
        List of tau values to try. If None, uses default values
    n_splits : int
        Number of cross-validation folds
    random_state : int
        Random seed
    
    Returns:
    --------
    pd.DataFrame
        Results dataframe
    dict
        Best parameters
    """
    # Default tau values if not provided
    if tau_values is None:
        tau_values = [0.1, 0.3, 0.5, 0.7, 1.0]
    
    # Load dataset
    X, y = load_dataset(dataset_name, dataset_type)
    
    if X.size == 0 or y.size == 0:
        print(f"Failed to load dataset {dataset_name} ({dataset_type})")
        return None, None
        
    print(f"Tuning tau parameter for {dataset_name} ({dataset_type}) dataset")
    print(f"Dataset shape: {X.shape}, Testing tau values: {tau_values}")
    
    # Setup cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Results storage
    results = []
    
    for tau in tau_values:
        print(f"  Testing tau={tau}")
        cv_metrics = {
            'accuracy': [],
            'auc': [],
            'train_time': [],
            'num_features': []
        }
        
        # Cross-validation
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            print(f"    Fold {fold_idx+1}/{n_splits}")
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Standardize data
            X_train, X_test, _ = standardize_data(X_train, X_test)
            
            try:
                # Train model
                model = PinFSSVM(B=B, C=C, tau=tau)
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_test)
                y_score = model.decision_function(X_test)
                
                # Evaluate
                metrics = evaluate_model(y_test, y_pred, y_score)
                cv_metrics['accuracy'].append(metrics['accuracy'])
                cv_metrics['auc'].append(metrics['auc'])
                cv_metrics['train_time'].append(model.train_time)
                cv_metrics['num_features'].append(model.get_num_selected_features())
            except Exception as e:
                print(f"    Error in fold {fold_idx+1}: {e}")
        
        # Skip if no results for this tau value
        if not cv_metrics['accuracy']:
            print(f"  No valid results for tau={tau}")
            continue
            
        # Aggregate results for this tau value
        result = {
            'tau': tau,
            'accuracy_mean': np.mean(cv_metrics['accuracy']),
            'accuracy_std': np.std(cv_metrics['accuracy']),
            'auc_mean': np.mean(cv_metrics['auc']),
            'auc_std': np.std(cv_metrics['auc']),
            'train_time_mean': np.mean(cv_metrics['train_time']),
            'num_features_mean': np.mean(cv_metrics['num_features'])
        }
        results.append(result)
        print(f"  tau={tau}: Accuracy={result['accuracy_mean']:.4f}, AUC={result['auc_mean']:.4f}")
    
    # Check if we have any results
    if not results:
        print("No valid results were obtained during parameter tuning")
        return None, None
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Find best tau value
    best_idx = results_df['auc_mean'].idxmax()
    best_params = {
        'tau': results_df.iloc[best_idx]['tau'],
        'auc_mean': results_df.iloc[best_idx]['auc_mean']
    }
    
    print(f"Best tau value: {best_params['tau']} with AUC: {best_params['auc_mean']:.4f}")
    
    return results_df, best_params

def plot_parameter_tuning(results_df, param_name, metric='auc_mean', title=None, figsize=(10, 6), output_dir='results'):
    """
    Plot parameter tuning results
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Results dataframe from tuning
    param_name : str
        Parameter name (e.g., 'C' or 'tau')
    metric : str
        Metric to plot
    title : str or None
        Plot title
    figsize : tuple
        Figure size
    output_dir : str
        Directory to save plot
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    if results_df is None or results_df.empty:
        print("No results to plot")
        return None
        
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=figsize)
    
    plt.plot(results_df[param_name], results_df[metric], marker='o', linestyle='-')
    
    # Add error bars if standard deviation is available
    std_col = metric.replace('_mean', '_std')
    if std_col in results_df.columns:
        plt.fill_between(
            results_df[param_name],
            results_df[metric] - results_df[std_col],
            results_df[metric] + results_df[std_col],
            alpha=0.2
        )
    
    # Highlight best point
    best_idx = results_df[metric].idxmax()
    best_x = results_df.iloc[best_idx][param_name]
    best_y = results_df.iloc[best_idx][metric]
    plt.scatter(best_x, best_y, color='red', s=100, zorder=10)
    plt.annotate(f'Best: {best_x}', (best_x, best_y), 
                 xytext=(0, 10), textcoords='offset points',
                 ha='center', fontsize=12)
    
    plt.xlabel(param_name, fontsize=12)
    plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
    
    if param_name == 'C' and len(results_df) > 2 and np.max(results_df[param_name]) / np.min(results_df[param_name]) > 10:
        plt.xscale('log')
    
    if title:
        plt.title(title, fontsize=14)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the figure
    save_path = os.path.join(output_dir, f'{param_name}_tuning_{metric}.png')
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    
    return plt.gcf()

def tune_parameters_grid(dataset_name, dataset_type="original", B=None,
                       C_values=None, tau_values=None, n_splits=5, random_state=42,
                       output_dir='results'):
    """
    Perform grid search for C and tau parameters
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset
    dataset_type : str
        Type of dataset ('original', 'noise', 'outlier', 'both')
    B : int or None
        Number of features to select
    C_values : list or None
        List of C values to try. If None, uses default range
    tau_values : list or None
        List of tau values to try. If None, uses default values
    n_splits : int
        Number of cross-validation folds
    random_state : int
        Random seed
    output_dir : str
        Directory to save results
    
    Returns:
    --------
    pd.DataFrame
        Results dataframe
    dict
        Best parameters
    """
    # Default parameter values
    if C_values is None:
        C_values = [0.01, 0.1, 1, 10, 100]
    if tau_values is None:
        tau_values = [0.1, 0.3, 0.5, 0.7, 1.0]
    
    # Load dataset
    X, y = load_dataset(dataset_name, dataset_type)
    
    if X.size == 0 or y.size == 0:
        print(f"Failed to load dataset {dataset_name} ({dataset_type})")
        return None, None
        
    print(f"Grid search for {dataset_name} ({dataset_type}) dataset")
    print(f"Dataset shape: {X.shape}")
    print(f"Testing C values: {C_values}")
    print(f"Testing tau values: {tau_values}")
    
    # Setup cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Results storage
    all_results = []
    
    for C in C_values:
        for tau in tau_values:
            print(f"  Testing C={C}, tau={tau}")
            cv_metrics = {
                'accuracy': [],
                'auc': [],
                'train_time': [],
                'num_features': []
            }
            
            # Cross-validation
            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Standardize data
                X_train, X_test, _ = standardize_data(X_train, X_test)
                
                try:
                    # Train model
                    model = PinFSSVM(B=B, C=C, tau=tau)
                    model.fit(X_train, y_train)
                    
                    # Predict
                    y_pred = model.predict(X_test)
                    y_score = model.decision_function(X_test)
                    
                    # Evaluate
                    metrics = evaluate_model(y_test, y_pred, y_score)
                    cv_metrics['accuracy'].append(metrics['accuracy'])
                    cv_metrics['auc'].append(metrics['auc'])
                    cv_metrics['train_time'].append(model.train_time)
                    cv_metrics['num_features'].append(model.get_num_selected_features())
                except Exception as e:
                    print(f"    Error in fold {fold_idx+1}: {e}")
            
            # Skip if no results for this parameter combination
            if not cv_metrics['accuracy']:
                print(f"  No valid results for C={C}, tau={tau}")
                continue
                
            # Aggregate results for this parameter combination
            result = {
                'C': C,
                'tau': tau,
                'accuracy_mean': np.mean(cv_metrics['accuracy']),
                'accuracy_std': np.std(cv_metrics['accuracy']),
                'auc_mean': np.mean(cv_metrics['auc']),
                'auc_std': np.std(cv_metrics['auc']),
                'train_time_mean': np.mean(cv_metrics['train_time']),
                'num_features_mean': np.mean(cv_metrics['num_features'])
            }
            all_results.append(result)
            print(f"  C={C}, tau={tau}: Accuracy={result['accuracy_mean']:.4f}, AUC={result['auc_mean']:.4f}")
    
    # Check if we have any results
    if not all_results:
        print("No valid results were obtained during grid search")
        return None, None
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Find best parameter combination
    best_idx = results_df['auc_mean'].idxmax()
    best_params = {
        'C': results_df.iloc[best_idx]['C'],
        'tau': results_df.iloc[best_idx]['tau'],
        'auc_mean': results_df.iloc[best_idx]['auc_mean']
    }
    
    print(f"Best parameters: C={best_params['C']}, tau={best_params['tau']} with AUC={best_params['auc_mean']:.4f}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, f'grid_search_{dataset_name}_{dataset_type}.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    
    # Create heatmap if we have enough results
    if len(tau_values) > 1 and len(C_values) > 1:
        try:
            pivot_table = results_df.pivot_table(values='auc_mean', index='C', columns='tau')
            
            plt.figure(figsize=(10, 8))
            im = plt.imshow(pivot_table, cmap='viridis', aspect='auto', interpolation='nearest', origin='lower')
            plt.colorbar(im, label='AUC Mean')
            
            # Set x and y ticks
            plt.xticks(range(len(pivot_table.columns)), pivot_table.columns)
            plt.yticks(range(len(pivot_table.index)), pivot_table.index)
            
            plt.xlabel('Tau', fontsize=12)
            plt.ylabel('C', fontsize=12)
            plt.title(f'Parameter Grid Search: {dataset_name} ({dataset_type})', fontsize=14)
            
            # Highlight best combination
            best_C_idx = np.where(pivot_table.index == best_params['C'])[0][0]
            best_tau_idx = np.where(pivot_table.columns == best_params['tau'])[0][0]
            plt.scatter(best_tau_idx, best_C_idx, marker='*', color='red', s=200)
            
            heatmap_path = os.path.join(output_dir, f'grid_heatmap_{dataset_name}_{dataset_type}.png')
            plt.tight_layout()
            plt.savefig(heatmap_path)
            print(f"Heatmap saved to {heatmap_path}")
        except Exception as e:
            print(f"Error creating heatmap: {e}")
    
    return results_df, best_params

if __name__ == "__main__":
    # Example usage
    dataset_name = 'wdbc'
    dataset_type = 'original'
    
    # Create output directory
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Tune C parameter
    c_results, c_best = tune_C_parameter(
        dataset_name=dataset_name, 
        dataset_type=dataset_type,
        B=5,
        tau=0.5
    )
    
    if c_results is not None:
        plt = plot_parameter_tuning(
            c_results, 
            'C', 
            title=f'C Parameter Tuning: {dataset_name} ({dataset_type})',
            output_dir=results_dir
        )
    
    # Tune tau parameter
    best_C = c_best['C'] if c_best else 1.0
    tau_results, tau_best = tune_tau_parameter(
        dataset_name=dataset_name, 
        dataset_type=dataset_type,
        B=5,
        C=best_C
    )
    
    if tau_results is not None:
        plt = plot_parameter_tuning(
            tau_results, 
            'tau', 
            title=f'Tau Parameter Tuning: {dataset_name} ({dataset_type})',
            output_dir=results_dir
        )
    
    # Grid search
    grid_results, grid_best = tune_parameters_grid(
        dataset_name=dataset_name, 
        dataset_type=dataset_type,
        B=5,
        output_dir=results_dir
    )
