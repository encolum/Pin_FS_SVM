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
    Vẽ biểu đồ cho nhiều dataset_type trên cùng một biểu đồ
    results_dict: Dictionary với key là dataset_type và value là DataFrame chứa kết quả
    """
    if not results_dict or all(df is None or df.empty for df in results_dict.values()):
        print("Không có kết quả để vẽ")
        return None
    if param_name == 'tau':
        param_name = r"$\tau$"
    os.makedirs(output_dir, exist_ok=True) 
    
    plt.figure(figsize=figsize)
    
    # Màu và marker cho mỗi dataset_type
    colors = {'original': 'blue', 'noise': 'red', 'outlier': 'green', 'both': 'purple'}
    markers = {'original': 'o', 'noise': 's', 'outlier': '^', 'both': 'D'}
    legend_handles = []
    legend_texts = []
    
    for dataset_type in dataset_types:
        if dataset_type not in results_dict or results_dict[dataset_type] is None or results_dict[dataset_type].empty:
            continue
            
        plot_df = results_dict[dataset_type].sort_values(by=param_name)
        
        # Vẽ đường cho dataset_type hiện tại
        line, = plt.plot(plot_df[param_name], plot_df[metric], 
                marker=markers.get(dataset_type, 'o'),
                linestyle='-', 
                color=colors.get(dataset_type, 'black'),
                )
        legend_handles.append(line)
        
        # Thêm vùng độ lệch chuẩn nếu có
        std_col = metric.replace('_mean', '_std')
        if std_col in plot_df.columns:
            plt.fill_between(
                plot_df[param_name],
                plot_df[metric] - plot_df[std_col],
                plot_df[metric] + plot_df[std_col],
                alpha=0.05,
                color=colors.get(dataset_type, 'black')
            )
        legend_text_for_type = f"{dataset_type.title()}"
        # Đánh dấu điểm tốt nhất cho mỗi dataset_type
        if not plot_df[metric].empty:
            try:
                best_idx = plot_df[metric].idxmax() 
                best_x = plot_df.loc[best_idx, param_name]
                best_y = plot_df.loc[best_idx, metric]
                plt.scatter(best_x, best_y, color=colors.get(dataset_type, 'black'), 
                           s=120, zorder=10, edgecolor='black',marker=markers.get(dataset_type, 'o')
                           )
                legend_text_for_type += f" - Best {param_name}={best_x:.2g} (AUC={best_y:.4f})"
            except ValueError: 
                print(f"Không tìm thấy điểm tốt nhất cho metric '{metric}' với {dataset_type}")
        legend_texts.append(legend_text_for_type)
        
    
    plt.xlabel(str(param_name).replace('_', ' ').title(), fontsize=12) 
    plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
    
    # Đặt thang logarit nếu cần
    if str(param_name).upper() == 'C': 
        plt.xscale('log', base=2)
        plt.grid(True, which='both', axis='x', linestyle='--', alpha=0.7)
    elif str(param_name).upper() == 'B':
        # Luôn sử dụng thang logarit cho B để hiển thị tốt hơn
        plt.xscale('log')
        
        # Lấy tất cả giá trị B xuất hiện trong dữ liệu để đặt làm tick
        all_b_values = set()
        for df in results_dict.values():
            if df is not None and not df.empty:
                all_b_values.update(df[param_name].unique())
        
        # Giới hạn số lượng tick để không quá đông
        all_b_values = sorted(all_b_values)
        if len(all_b_values) > 15:  # Nếu có quá nhiều giá trị
            # Chọn khoảng 10-15 giá trị đại diện
            if max(all_b_values) / min(all_b_values) > 100:
                # Nếu dải giá trị rộng, sử dụng thang logarit để chọn mốc
                log_ticks = np.geomspace(min(all_b_values), max(all_b_values), num=12)
                tick_positions = [closest_value(all_b_values, x) for x in log_ticks]
            else:
                # Nếu dải giá trị hẹp, chọn đều các mốc
                step = max(1, len(all_b_values) // 12)
                tick_positions = all_b_values[::step]
                # Luôn bao gồm giá trị đầu và cuối
                if all_b_values[0] not in tick_positions:
                    tick_positions = [all_b_values[0]] + tick_positions
                if all_b_values[-1] not in tick_positions:
                    tick_positions.append(all_b_values[-1])
        else:
            # Nếu ít giá trị, hiển thị tất cả
            tick_positions = all_b_values
        
        # Đặt tick positions và labels
        plt.xticks(tick_positions, [str(int(x)) for x in tick_positions])
        
        # Thêm lưới dọc tại các điểm mốc B
        plt.grid(True, which='both', axis='x', linestyle='--', alpha=0.7)
    else:
        # Cho các tham số khác
        plt.grid(True, linestyle='--', alpha=0.7)

    plot_title = title if title else f"So sánh độ nhạy của {metric} với {param_name} qua các dataset type"
    plt.title(plot_title, fontsize=14)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc = 'best', fontsize=10)
    if legend_handles and legend_texts: # Chỉ tạo legend nếu có gì để hiển thị
        plt.legend(legend_handles, legend_texts, fontsize=9, loc='best')
    plt.tight_layout()
    
    # Tạo tên file an toàn
    safe_title = f'comparison_{param_name}_across_types.png'
    save_path = os.path.join(output_dir, safe_title)
    
    try:
        plt.savefig(save_path, dpi = 300)
        print(f"Đã lưu biểu đồ vào {save_path}")
    except Exception as e:
        print(f"Lỗi khi lưu biểu đồ: {e}")
            
    plt.close() 
    return plt.gcf()

def closest_value(values, target):
    """Tìm giá trị gần nhất với target trong list values"""
    return min(values, key=lambda x: abs(x - target))

def generate_comparison_plots(model_class, dataset_name, dataset_types, best_params_dict, base_output_dir):
    """
    Tạo biểu đồ so sánh các dataset_type khác nhau
    
    Parameters:
    -----------
    model_class: class, lớp mô hình (PinFSSVM)
    dataset_name: str, tên dataset
    dataset_types: list, danh sách các dataset_type cần so sánh (original, noise, outlier, both)
    best_params_dict: dict, dictionary với key là dataset_type và value là dict chứa tham số tốt nhất
    base_output_dir: str, thư mục đầu ra
    """
    # Tạo thư mục đầu ra
    plot_output_dir = os.path.join(base_output_dir, "param_comparison_plots", dataset_name, model_class.__name__)
    os.makedirs(plot_output_dir, exist_ok=True)

    # Tạo K-fold cross-validation 
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    # Dictionary để lưu dữ liệu cho mỗi tham số
    tau_results_dict = {}
    C_results_dict = {}
    B_results_dict = {}
    
    # Định nghĩa giá trị tham số cần quét
    TAU_PLOT_VALUES = [0.1, 0.5, 1.0]
    C_PLOT_VALUES = [2**i for i in range(-3, 6)]  # 2^-3 đến 2^5
    
    # Tải dữ liệu cho mỗi dataset_type và tạo biểu đồ
    print(f"Đang tạo biểu đồ so sánh cho dataset {dataset_name} với các loại: {', '.join(dataset_types)}")
    
    # Dictionary lưu dữ liệu X, y của mỗi dataset_type để tránh tải lại nhiều lần
    datasets = {}
    
    # Tải dữ liệu cho mỗi dataset_type
    for dataset_type in dataset_types:
        print(f"  Đang tải dữ liệu cho {dataset_name} (type: {dataset_type})...")
        try:
            X, y = load_dataset(dataset_name, dataset_type)
            if X.shape[0] > 0 and X.shape[1] > 0:
                datasets[dataset_type] = (X, y)
                
                # Xác định giá trị B dựa trên số lượng features
                max_features = X.shape[1]
                best_B = best_params_dict.get('B')
                B_PLOT_VALUES = determine_B_values(max_features, best_B)
                
                # Lấy tham số tốt nhất
                if dataset_type in best_params_dict:
                    best_params = best_params_dict[dataset_type]
                    
                    # --- Quét tau với C và B cố định ---
                    print(f"  Đang tạo dữ liệu tau cho {dataset_type}...")
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
                    
                    # --- Quét C với tau và B cố định ---
                    print(f"  Đang tạo dữ liệu C cho {dataset_type}...")
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
                    
                    # --- Quét B với C và tau cố định ---
                    print(f"  Đang tạo dữ liệu B cho {dataset_type}...")
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
                print(f"  Dữ liệu rỗng cho {dataset_name} ({dataset_type})")
                
        except Exception as e:
            print(f"  Lỗi khi tải hoặc xử lý dữ liệu {dataset_name} ({dataset_type}): {e}")
    
    # Vẽ biểu đồ so sánh các dataset_type
    if tau_results_dict:
        print("Đang vẽ biểu đồ so sánh tau...")
        plot_parameter_tuning_multi_type(
            tau_results_dict, 'tau', dataset_types, metric='auc_mean', 
            title=f'So sánh AUC vs r"$\tau$" cho {dataset_name} qua các dataset type', 
            output_dir=plot_output_dir
        )
    
    if C_results_dict:
        print("Đang vẽ biểu đồ so sánh C...")
        plot_parameter_tuning_multi_type(
            C_results_dict, 'C', dataset_types, metric='auc_mean', 
            title=f'So sánh AUC vs C cho {dataset_name} qua các dataset type', 
            output_dir=plot_output_dir
        )
    
    if B_results_dict:
        print("Đang vẽ biểu đồ so sánh B...")
        plot_parameter_tuning_multi_type(
            B_results_dict, 'B', dataset_types, metric='auc_mean', 
            title=f'So sánh AUC vs B cho {dataset_name} qua các dataset type', 
            output_dir=plot_output_dir
        )
        
    print(f"Hoàn thành vẽ biểu đồ so sánh cho {dataset_name}")

def determine_B_values(max_features, best_B=None):
    """Hàm xác định giá trị B dựa trên số lượng đặc trưng với nhiều điểm hơn"""
    _candidate_B_values = []
    
    # Luôn bắt đầu với giá trị B=1
    _candidate_B_values.append(1)
    
    if max_features <= 0:
        return [1]
    elif max_features == 1:
        return [1]
    elif max_features <= 15:
        # Đối với dataset nhỏ, dùng tất cả các giá trị nguyên 
        _candidate_B_values = list(range(1, max_features + 1))
    elif max_features <= 35:
        # Thêm B=2 và B=3 để tăng mật độ điểm ở vùng đầu
        _candidate_B_values.extend([2, 3])
        # Thêm điểm ở giữa dải giá trị
        _candidate_B_values.extend(list(range(min(5, max_features), max_features + 1, max(1, max_features // 10))))
        # Thêm các điểm theo phần trăm
        for p in [0.2, 0.4, 0.6, 0.8]:
            b_val = int(round(p * max_features))
            if b_val >= 1:
                _candidate_B_values.append(b_val)
    elif max_features <= 70:
        # Thêm nhiều điểm hơn ở vùng đầu
        _candidate_B_values.extend([2, 3, 5, 7])
        # Điểm ở vùng giữa
        _candidate_B_values.extend(list(range(min(10, max_features), max_features + 1, max(1, max_features // 12))))
        # Thêm các điểm theo phần trăm
        for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            b_val = int(round(p * max_features))
            if b_val >= 1:
                _candidate_B_values.append(b_val)
    else:
        # Với dataset lớn, cần nhiều điểm ở vùng đầu
        _candidate_B_values.extend([2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50])
        
        # Thêm nhiều điểm phần trăm hơn
        percent_points = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        for p in percent_points:
            b_val = int(round(p * max_features))
            if b_val >= 1:
                _candidate_B_values.append(b_val)
        
        # Thêm điểm theo thang logarit
        if max_features > 100:
            # Tăng số lượng điểm logarit lên 8
            log_points = np.geomspace(min(20, max_features-1), 
                                      max_features-1 if max_features > 1 else 1, 
                                      num=8, dtype=int)
            _candidate_B_values.extend(log_points.tolist())

    # Luôn bao gồm giá trị max_features
    _candidate_B_values.append(max_features)
    
    # Đảm bảo best_B được bao gồm nếu cung cấp
    if best_B is not None and 1 <= best_B <= max_features and best_B not in _candidate_B_values:
        _candidate_B_values.append(best_B)
        print(f"Thêm best_B={best_B} từ file Excel vào danh sách quét B")
    
    # Loại bỏ trùng lặp và sắp xếp
    return sorted(list(set(b for b in _candidate_B_values if 1 <= b <= max_features)))

def main_compare_dataset_types():
    parser = argparse.ArgumentParser(description="So sánh hiệu suất giữa các loại dataset cho PinFSSVM.")
    parser.add_argument(
        "--results_file",
        type=str,
        required=True,
        help="Đường dẫn đến file Excel kết quả thí nghiệm."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Thư mục cơ sở để lưu biểu đồ."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Tên dataset cần so sánh."
    )
    parser.add_argument(
        "--dataset_types",
        type=str,
        nargs='+',
        default=['original', 'noise', 'outlier', 'both'],
        help="Danh sách các dataset_type cần so sánh."
    )

    args = parser.parse_args()

    print(f"Đang tải kết quả từ: {args.results_file}")
    if not os.path.exists(args.results_file):
        print(f"Lỗi: Không tìm thấy file kết quả tại {args.results_file}")
        return

    results_df_all = pd.read_excel(args.results_file)
    results_df_all.columns = results_df_all.columns.str.strip()  
    print(f"Đã tải {len(results_df_all)} kết quả từ Excel.")

    target_model_name = PinFSSVM.__name__
    expected_type_of_model_in_csv = "Not noise"  # Giữ nguyên như code gốc
    
    # Lọc kết quả cho dataset được chỉ định
    dataset_results = results_df_all[
        (results_df_all['Model'] == target_model_name) 
        # (results_df_all['Type of model'] == expected_type_of_model_in_csv) 
        # (results_df_all['Dataset'] == args.dataset_name)
    ]
    
    if dataset_results.empty:
        print(f"Không tìm thấy kết quả cho dataset '{args.dataset_name}' với model '{target_model_name}'.")
        return
    
    type_mapping = {
        'Not noise': 'original',
        'Noise': 'noise',
        'Outlier': 'outlier',
        'Noise + Outlier': 'both'
    }
    # Thu thập tham số tốt nhất cho mỗi dataset_type
    best_params_dict = {}
    for excel_type, code_type in type_mapping.items():
        if code_type in args.dataset_types:  # Chỉ xử lý các loại dataset được yêu cầu
            type_results = dataset_results[dataset_results['Type of model'] == excel_type]
            if not type_results.empty:
                # Lấy hàng có Average AUC cao nhất
                if 'Average AUC' in type_results.columns:
                    best_row = type_results.loc[type_results['Average AUC'].idxmax()]
                else:
                    # Nếu không có cột Average AUC, lấy hàng đầu tiên
                    best_row = type_results.iloc[0]
                
                params = {}
                for param in ['C', 'B', 'tau']:
                    if param in best_row.index:
                        params[param] = best_row[param]
                
                if params:
                    best_params_dict[code_type] = params
                    print(f"Tham số tốt nhất cho {code_type} ('{excel_type}' trong Excel): {params}")
    
    if not best_params_dict:
        print(f"Không tìm thấy tham số tốt nhất cho bất kỳ dataset_type nào.")
        return
    
    print(f"Đang tạo biểu đồ so sánh cho dataset {args.dataset_name} với các loại: {', '.join(args.dataset_types)}")
    try:
        generate_comparison_plots(
            model_class=PinFSSVM,
            dataset_name=args.dataset_name,
            dataset_types=args.dataset_types,
            best_params_dict=best_params_dict,
            base_output_dir=args.output_dir
        )
        print(f"Hoàn thành tạo biểu đồ so sánh cho {args.dataset_name}.")
    except Exception as e:
        print(f"Lỗi khi tạo biểu đồ so sánh: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main_compare_dataset_types()