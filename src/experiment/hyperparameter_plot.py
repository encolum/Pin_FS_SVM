import os
import time 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold
import sys
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.experiment.run_experiment import run_cv_for_params 
from src.utils.data_loader import load_dataset
from src.models.pin_fs_svm import PinFSSVM 



def plot_parameter_tuning_multi_type(results_dict, param_name, dataset_types, metric='auc_mean', 
                                    title=None, figsize=(12, 8), output_dir='results/plots'):
    """
    Plot charts for multiple dataset types on the same chart
    results_dict: Dictionary with key as dataset_type and value as DataFrame containing results
    """
    if not results_dict or all(df is None or df.empty for df in results_dict.values()):
        print("No results to plot")
        return None
    original_param_name = param_name
    display_param_name = param_name
    if param_name =='tau':
        x_axis_label = r"$\tau$"
        display_param_name = 'tau'
    else:
        x_axis_label = str(param_name).replace('_', ' ').title()
        display_param_name = x_axis_label
    os.makedirs(output_dir, exist_ok=True) 
    
    plt.figure(figsize=figsize)
    
    # Colors and markers for each dataset_type
    colors = {'original': 'blue', 'noise': 'red', 'outlier': 'green', 'both': 'purple'}
    markers = {'original': 'o', 'noise': 's', 'outlier': '^', 'both': 'D'}
    legend_handles = []
    legend_texts = []
    
    for dataset_type in dataset_types:
        if dataset_type not in results_dict or results_dict[dataset_type] is None or results_dict[dataset_type].empty:
            continue
            
        plot_df = results_dict[dataset_type].sort_values(by=original_param_name)
        
        # Draw line for current dataset_type
        line, = plt.plot(plot_df[original_param_name], plot_df[metric], 
                marker=markers.get(dataset_type, 'o'),
                linestyle='-', 
                color=colors.get(dataset_type, 'black'),
                )
        legend_handles.append(line)
        
        # Add standard deviation area if available
        std_col = metric.replace('_mean', '_std')
        if std_col in plot_df.columns:
            plt.fill_between(
                plot_df[original_param_name],
                plot_df[metric] - plot_df[std_col],
                plot_df[metric] + plot_df[std_col],
                alpha=0.05,
                color=colors.get(dataset_type, 'black')
            )
        legend_text_for_type = f"{dataset_type.title()}"
        # Mark the best point for each dataset_type
        if not plot_df[metric].empty:
            try:
                best_idx = plot_df[metric].idxmax() 
                best_x = plot_df.loc[best_idx, original_param_name]
                best_y = plot_df.loc[best_idx, metric]
                plt.scatter(best_x, best_y, color=colors.get(dataset_type, 'black'), 
                           s=120, zorder=10, edgecolor='black',marker=markers.get(dataset_type, 'o')
                           )
                legend_text_for_type += f" - Best {x_axis_label}={best_x:.2g} (AUC={best_y:.4f})"
            except ValueError: 
                print(f"Could not find best point for metric '{metric}' with {dataset_type}")
        legend_texts.append(legend_text_for_type)
        
    
    plt.xlabel(x_axis_label, fontsize=12) 
    plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
    
    # Set logarithmic scale if needed
    if str(original_param_name).upper() == 'C': 
        plt.xscale('log', base=2)
        plt.grid(True, which='both', axis='x', linestyle='--', alpha=0.7)
    elif str(original_param_name).upper() == 'B':
        # Always use logarithmic scale for B for better display
        plt.xscale('log')
        
        # Get all B values in the data to set as ticks
        all_b_values = set()
        for df in results_dict.values():
            if df is not None and not df.empty:
                all_b_values.update(df[original_param_name].unique())
        
        # Limit the number of ticks to avoid overcrowding
        all_b_values = sorted(all_b_values)
        if len(all_b_values) > 15:  # If there are too many values
            # Choose about 10-15 representative values
            if max(all_b_values) / min(all_b_values) > 100:
                # If range is wide, use logarithmic scale for markers
                log_ticks = np.geomspace(min(all_b_values), max(all_b_values), num=12)
                tick_positions = [closest_value(all_b_values, x) for x in log_ticks]
            else:
                # If range is narrow, evenly space the markers
                step = max(1, len(all_b_values) // 12)
                tick_positions = all_b_values[::step]
                # Always include first and last values
                if all_b_values[0] not in tick_positions:
                    tick_positions = [all_b_values[0]] + tick_positions
                if all_b_values[-1] not in tick_positions:
                    tick_positions.append(all_b_values[-1])
        else:
            # If few values, display all
            tick_positions = all_b_values
        
        # Set tick positions and labels
        plt.xticks(tick_positions, [str(int(x)) for x in tick_positions])
        
        # Add vertical grid at B marker points
        plt.grid(True, which='both', axis='x', linestyle='--', alpha=0.7)
    else:
        # For other parameters
        plt.grid(True, linestyle='--', alpha=0.7)

    plot_title = title if title else f"Sensitivity comparison of {metric} to {display_param_name} across dataset types"
    plt.title(plot_title, fontsize=14)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    if legend_handles and legend_texts: 
        plt.legend(legend_handles, legend_texts, fontsize=10, loc='best')
    plt.tight_layout()
    
    # Create safe filename
    safe_title = f'comparison_{param_name}_across_types.png'
    save_path = os.path.join(output_dir, safe_title)
    
    try:
        plt.savefig(save_path, dpi = 300)
        print(f"Chart saved to {save_path}")
    except Exception as e:
        print(f"Error when saving chart: {e}")
            
    plt.close() 
    return plt.gcf()

def closest_value(values, target):
    """Find the closest value to the target in the values list"""
    return min(values, key=lambda x: abs(x - target))

def generate_comparison_plots(model_class, dataset_name, dataset_types, best_params_dict, base_output_dir):
    """
    Generate comparison charts for different dataset types
    
    Parameters:
    -----------
    model_class: class, model class (PinFSSVM)
    dataset_name: str, dataset name
    dataset_types: list, list of dataset types to compare (original, noise, outlier, both)
    best_params_dict: dict, dictionary with key as dataset_type and value as dict containing best parameters
    base_output_dir: str, output directory
    """
    # Create output directory
    plot_output_dir = os.path.join(base_output_dir, "param_comparison_plots", dataset_name, model_class.__name__)
    os.makedirs(plot_output_dir, exist_ok=True)

    # Create K-fold cross-validation 
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    # Dictionary to store data for each parameter
    tau_results_dict = {}
    C_results_dict = {}
    B_results_dict = {}
    
    # Define parameter values to scan
    TAU_PLOT_VALUES = [0.1, 0.5, 1.0]
    C_PLOT_VALUES = [2**i for i in range(-3, 6)]  # 2^-3 to 2^5
    
    # Load data for each dataset_type and create charts
    print(f"Creating comparison charts for dataset {dataset_name} with types: {', '.join(dataset_types)}")
    
    # Dictionary to store X, y data for each dataset_type to avoid reloading
    datasets = {}
    
    # Load data for each dataset_type
    for dataset_type in dataset_types:
        print(f"  Loading data for {dataset_name} (type: {dataset_type})...")
        try:
            X, y = load_dataset(dataset_name, dataset_type)
            if X.shape[0] > 0 and X.shape[1] > 0:
                datasets[dataset_type] = (X, y)
                
                # Determine B values based on number of features
                max_features = X.shape[1]
                best_B = best_params_dict.get('B')
                B_PLOT_VALUES = determine_B_values(max_features, best_B)
                
                # Get best parameters
                if dataset_type in best_params_dict:
                    best_params = best_params_dict[dataset_type]
                    
                    # --- Scan tau with fixed C and B ---
                    print(f"  Generating tau data for {dataset_type}...")
                    best_C = best_params.get('C', 1.0)
                    best_B = best_params.get('B', max_features//2 if max_features > 1 else 1)
                    results_tau = []
                    
                    for tau_val in TAU_PLOT_VALUES:
                        current_params = {'C': best_C, 'B': best_B, 'tau': tau_val}
                        for p_name, p_val in best_params.items():
                            if p_name not in current_params:
                                current_params[p_name] = p_val
                        
                        cv_result = run_cv_for_params(model_class, current_params, X, y, kf)
                        if cv_result and cv_result['metrics']['auc']:
                            mean_auc = np.mean(cv_result['metrics']['auc'])
                            std_auc = np.std(cv_result['metrics']['auc'])
                            results_tau.append({'tau': tau_val, 'auc_mean': mean_auc, 'auc_std': std_auc})
                    
                    if results_tau:
                        tau_results_dict[dataset_type] = pd.DataFrame(results_tau)
                    
                    # --- Scan C with fixed tau and B ---
                    print(f"  Generating C data for {dataset_type}...")
                    best_tau = best_params.get('tau', 0.5)
                    results_C = []
                    
                    for c_val in C_PLOT_VALUES:
                        current_params = {'C': c_val, 'B': best_B, 'tau': best_tau}
                        for p_name, p_val in best_params.items():
                            if p_name not in current_params:
                                current_params[p_name] = p_val
                        
                        cv_result = run_cv_for_params(model_class, current_params, X, y, kf)
                        if cv_result and cv_result['metrics']['auc']:
                            mean_auc = np.mean(cv_result['metrics']['auc'])
                            std_auc = np.std(cv_result['metrics']['auc'])
                            results_C.append({'C': c_val, 'auc_mean': mean_auc, 'auc_std': std_auc})
                    
                    if results_C:
                        C_results_dict[dataset_type] = pd.DataFrame(results_C)
                    
                    # --- Scan B with fixed C and tau ---
                    print(f"  Generating B data for {dataset_type}...")
                    results_B = []
                    
                    for b_val in B_PLOT_VALUES:
                        current_params = {'C': best_C, 'B': b_val, 'tau': best_tau}
                        for p_name, p_val in best_params.items():
                            if p_name not in current_params:
                                current_params[p_name] = p_val
                        
                        cv_result = run_cv_for_params(model_class, current_params, X, y, kf)
                        if cv_result and cv_result['metrics']['auc']:
                            mean_auc = np.mean(cv_result['metrics']['auc'])
                            std_auc = np.std(cv_result['metrics']['auc'])
                            avg_num_features = np.mean(cv_result['metrics']['num_features'])
                            results_B.append({'B': b_val, 'auc_mean': mean_auc, 'auc_std': std_auc, 'num_features_mean': avg_num_features})
                    
                    if results_B:
                        B_results_dict[dataset_type] = pd.DataFrame(results_B).dropna(subset=['auc_mean'])
            
            else:
                print(f"  Empty data for {dataset_name} ({dataset_type})")
                
        except Exception as e:
            print(f"  Error loading or processing data {dataset_name} ({dataset_type}): {e}")
    
    # Draw comparison charts for dataset types
    if tau_results_dict:
        print("Drawing tau comparison chart...")
        title = 'AUC vs $\\tau$ comparison for {} across dataset types'.format(dataset_name)
        plot_parameter_tuning_multi_type(
            tau_results_dict, 'tau', dataset_types, metric='auc_mean', 
            title=title,
            output_dir=plot_output_dir
        )
        
    if C_results_dict:
        print("Drawing C comparison chart...")
        plot_parameter_tuning_multi_type(
            C_results_dict, 'C', dataset_types, metric='auc_mean', 
            title=f'AUC vs C comparison for {dataset_name} across dataset types', 
            output_dir=plot_output_dir
        )
    
    if B_results_dict:
        print("Drawing B comparison chart...")
        plot_parameter_tuning_multi_type(
            B_results_dict, 'B', dataset_types, metric='auc_mean', 
            title=f'AUC vs B comparison for {dataset_name} across dataset types', 
            output_dir=plot_output_dir
        )
        
    print(f"Completed drawing comparison charts for {dataset_name}")

def determine_B_values(max_features, best_B=None):
    """Function to determine B values based on the number of features with more points"""
    _candidate_B_values = []
    
    # Always start with B=1
    _candidate_B_values.append(1)
    
    if max_features <= 0:
        return [1]
    elif max_features == 1:
        return [1]
    elif max_features <= 15:
        # For small datasets, use all integer values 
        _candidate_B_values = list(range(1, max_features + 1))
    elif max_features <= 35:
        # Add B=2 and B=3 to increase density at the beginning
        _candidate_B_values.extend([2, 3])
        # Add points in the middle range
        _candidate_B_values.extend(list(range(min(5, max_features), max_features + 1, max(1, max_features // 10))))
        # Add points by percentage
        for p in [0.2, 0.4, 0.6, 0.8]:
            b_val = int(round(p * max_features))
            if b_val >= 1:
                _candidate_B_values.append(b_val)
    elif max_features <= 70:
        # Add more points at the beginning
        _candidate_B_values.extend([2, 3, 5, 7])
        # Points in the middle range
        _candidate_B_values.extend(list(range(min(10, max_features), max_features + 1, max(1, max_features // 12))))
        # Add points by percentage
        for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            b_val = int(round(p * max_features))
            if b_val >= 1:
                _candidate_B_values.append(b_val)
    else:
        # For large datasets, need more points at the beginning
        _candidate_B_values.extend([2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50])
        
        # Add more percentage points
        percent_points = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        for p in percent_points:
            b_val = int(round(p * max_features))
            if b_val >= 1:
                _candidate_B_values.append(b_val)
        
        # Add logarithmic points
        if max_features > 100:
            # Increase number of logarithmic points to 8
            log_points = np.geomspace(min(20, max_features-1), 
                                      max_features-1 if max_features > 1 else 1, 
                                      num=8, dtype=int)
            _candidate_B_values.extend(log_points.tolist())

    # Always include max_features
    _candidate_B_values.append(max_features)
    
    # Ensure best_B is included if provided
    if best_B is not None and 1 <= best_B <= max_features and best_B not in _candidate_B_values:
        _candidate_B_values.append(best_B)
        print(f"Adding best_B={best_B} from Excel file to B scan list")
    
    # Remove duplicates and sort
    return sorted(list(set(b for b in _candidate_B_values if 1 <= b <= max_features)))

def main_compare_dataset_types():
    parser = argparse.ArgumentParser(description="Compare performance between dataset types for PinFSSVM.")
    parser.add_argument(
        "--results_file",
        type=str,
        required=True,
        help="Path to the experiment results Excel file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Base directory to save charts."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Dataset name to compare."
    )
    parser.add_argument(
        "--dataset_types",
        type=str,
        nargs='+',
        default=['original', 'noise', 'outlier', 'both'],
        help="List of dataset types to compare."
    )

    args = parser.parse_args()

    print(f"Loading results from: {args.results_file}")
    if not os.path.exists(args.results_file):
        print(f"Error: Could not find results file at {args.results_file}")
        return

    results_df_all = pd.read_excel(args.results_file)
    results_df_all.columns = results_df_all.columns.str.strip()  
    print(f"Loaded {len(results_df_all)} results from Excel.")

    target_model_name = PinFSSVM.__name__
    expected_type_of_model_in_csv = "Not noise"  # Keep as in original code
    
    # Filter results for the specified dataset
    dataset_results = results_df_all[
        (results_df_all['Model'] == target_model_name) 
        # (results_df_all['Type of model'] == expected_type_of_model_in_csv) 
        # (results_df_all['Dataset'] == args.dataset_name)
    ]
    
    if dataset_results.empty:
        print(f"No results found for dataset '{args.dataset_name}' with model '{target_model_name}'.")
        return
    
    type_mapping = {
        'Not noise': 'original',
        'Noise': 'noise',
        'Outlier': 'outlier',
        'Noise + Outlier': 'both'
    }
    # Collect best parameters for each dataset_type
    best_params_dict = {}
    for excel_type, code_type in type_mapping.items():
        if code_type in args.dataset_types:  # Only process requested dataset types
            type_results = dataset_results[dataset_results['Type of model'] == excel_type]
            if not type_results.empty:
                # Get row with highest Average AUC
                if 'Average AUC' in type_results.columns:
                    best_row = type_results.loc[type_results['Average AUC'].idxmax()]
                else:
                    # If no Average AUC column, take first row
                    best_row = type_results.iloc[0]
                
                params = {}
                for param in ['C', 'B', 'tau']:
                    if param in best_row.index:
                        params[param] = best_row[param]
                
                if params:
                    best_params_dict[code_type] = params
                    print(f"Best parameters for {code_type} ('{excel_type}' in Excel): {params}")
    
    if not best_params_dict:
        print(f"Could not find best parameters for any dataset type.")
        return
    
    print(f"Creating comparison charts for dataset {args.dataset_name} with types: {', '.join(args.dataset_types)}")
    try:
        generate_comparison_plots(
            model_class=PinFSSVM,
            dataset_name=args.dataset_name,
            dataset_types=args.dataset_types,
            best_params_dict=best_params_dict,
            base_output_dir=args.output_dir
        )
        print(f"Completed creating comparison charts for {args.dataset_name}.")
    except Exception as e:
        print(f"Error creating comparison charts: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main_compare_dataset_types()