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
    # A - Load dataset
    X, y = load_dataset(dataset_name, dataset_type)
    if X.size == 0 or y.size == 0:
        print(f"Không thể load dataset {dataset_name} ({dataset_type})")
        return None

    print(f"Đang chạy {model_class.__name__} trên {dataset_name} ({dataset_type})")

    # B - Cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    accuracy_scores = []
    auc_scores = []
    f1_scores = []
    g_means = []
    train_times = []
    num_features_list = []
    all_selected_features = []

    for fold_idx, (train_index, test_index) in enumerate(kf.split(X)):
        print(f"  Fold {fold_idx+1}/{n_splits}")

        # C - Tách dữ liệu
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # D - Chuẩn hóa dữ liệu
        X_train, X_test, _ = standardize_data(X_train, X_test)

        try:
            # E - Huấn luyện mô hình
            model = model_class(**model_params)
            model.fit(X_train, y_train)

            # F - Dự đoán và đánh giá
            y_pred = model.predict(X_test)
            y_score = model.decision_function(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_score)
            f1 = f1_score(y_test, y_pred)
            g_mean = geometric_mean_score(y_test, y_pred)

            accuracy_scores.append(accuracy)
            auc_scores.append(auc)
            f1_scores.append(f1)
            g_means.append(g_mean)
            train_times.append(model.train_time)

            num_features, selected_features = count_selected_features(model.w)
            num_features_list.append(num_features)
            all_selected_features.append(selected_features)

        except Exception as e:
            print(f"    Lỗi ở fold {fold_idx+1}: {e}")

    # G - Tổng hợp kết quả
    if not accuracy_scores:
        print("Không có kết quả hợp lệ.")
        return None

    result = {
        'model': model_class.__name__,
        'dataset': dataset_name,
        'dataset_type': dataset_type,
        'params': model_params,
        'accuracy_mean': np.mean(accuracy_scores),
        'accuracy_std': np.std(accuracy_scores),
        'auc_mean': np.mean(auc_scores),
        'auc_std': np.std(auc_scores),
        'f1_score_mean': np.mean(f1_scores),
        'g_mean_mean': np.mean(g_means),
        'train_time_mean': np.mean(train_times),
        'num_features_mean': np.mean(num_features_list),
        'selected_features': all_selected_features,
        'stability': feature_selection_stability(all_selected_features, X.shape[1])
    }

    return result

    
def run_full_experiment(models_config, datasets_config, output_dir='results'):
    """
    Run a full experiment with multiple models and datasets
    
    Parameters:
    -----------
    models_config : list of dict
        Each dict contains model_class, model_params_grid (đối với tối ưu tham số)
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
        
        # Kiểm tra xem có phải là một tham số để tối ưu hay không
        if 'param_grid' in model_config:
            param_name = list(model_config['param_grid'].keys())[0]  # Lấy tên tham số (ví dụ: 'C')
            param_values = model_config['param_grid'][param_name]    # Lấy danh sách giá trị
            
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
                    n_splits = 10  # Số fold mặc định
                    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                    
                    best_param_value = None
                    best_mean_cv_accuracy = 0
                    best_mean_cv_AUC = 0
                    best_all_selected_features = []
                    
                    for param_value in param_values:
                        # Tạo dictionary chứa tham số hiện tại
                        current_params = {param_name: param_value}
                        
                        # Thêm các tham số cố định nếu có
                        if 'fixed_params' in model_config:
                            current_params.update(model_config['fixed_params'])
                        
                        # Đánh giá tham số hiện tại với cross-validation
                        cv_metrics = {
                            'accuracy': [],
                            'auc': [],
                            'f1_score': [],
                            'g_mean': [],
                            'train_time': [],
                            'num_features': []
                        }
                        cv_selected_features = []
                        
                        for train_idx, test_idx in kf.split(X):
                            X_train, X_test = X[train_idx], X[test_idx]
                            y_train, y_test = y[train_idx], y[test_idx]
                            
                            # Standardize data
                            X_train, X_test, _ = standardize_data(X_train, X_test)
                            
                            try:
                                # Huấn luyện mô hình với tham số hiện tại
                                model = model_class(**current_params)
                                model.fit(X_train, y_train)
                                
                                # Dự đoán
                                y_pred = model.predict(X_test)
                                y_score = model.decision_function(X_test)
                                
                                # Đánh giá
                                metrics = evaluate_model(y_test, y_pred, y_score)
                                for key in metrics:
                                    cv_metrics[key].append(metrics[key])
                                
                                # Lưu thời gian và các đặc trưng được chọn
                                cv_metrics['train_time'].append(model.train_time)
                                num_features, selected_features = count_selected_features(model.w)
                                cv_metrics['num_features'].append(num_features)
                                cv_selected_features.append(selected_features)
                                
                            except Exception as e:
                                print(f"    Error with {param_name}={param_value}: {e}")
                        
                        # Tính giá trị trung bình qua các fold
                        if cv_metrics['accuracy']:
                            mean_cv_accuracy = np.mean(cv_metrics['accuracy'])
                            mean_cv_AUC = np.mean(cv_metrics['auc'])
                            print(f"  {param_name}={param_value}: Accuracy={mean_cv_accuracy:.4f}, AUC={mean_cv_AUC:.4f}")
                            
                            # Cập nhật tham số tốt nhất nếu AUC cao hơn
                            if mean_cv_AUC > best_mean_cv_AUC:
                                best_param_value = param_value
                                best_mean_cv_accuracy = mean_cv_accuracy
                                best_mean_cv_AUC = mean_cv_AUC
                                best_cv_metrics = cv_metrics.copy()
                                best_all_selected_features = cv_selected_features
                    
                    # if best_param_value is not None:
                    #     # Tính độ ổn định của việc lựa chọn đặc trưng
                    #     stability = feature_selection_stability(best_all_selected_features, X.shape[1])
                        
                        # Tính tần suất xuất hiện của các đặc trưng
                        feature_freq = {}
                        for features in best_all_selected_features:
                            for f in features:
                                if f not in feature_freq:
                                    feature_freq[f] = 0
                                feature_freq[f] += 1
                        
                        # Các đặc trưng được chọn thường xuyên nhất
                        frequent_features = sorted([(f, freq/n_splits) for f, freq in feature_freq.items()], 
                                                 key=lambda x: x[1], reverse=True)
                        
                        # Kết quả
                        result = {
                            'model': model_class.__name__,
                            'dataset': dataset_name,
                            'dataset_type': dataset_type,
                            'params': {param_name: best_param_value},
                            # 'stability': stability
                        }
                        
                        # # Thêm các thông số về độ chính xác
                        # for key in best_cv_metrics:
                        #     result[f'{key}_mean'] = np.mean(best_cv_metrics[key])
                        #     result[f'{key}_std'] = np.std(best_cv_metrics[key])
                        
                        # Thêm thông tin về đặc trưng
                        result['frequent_features'] = frequent_features
                        result['all_selected_features'] = best_all_selected_features
                        
                        all_results.append(result)
                        print(f"  Best {param_name}={best_param_value}, Accuracy={best_mean_cv_accuracy:.4f}, AUC={best_mean_cv_AUC:.4f}")
    
    # Kiểm tra xem có kết quả nào không
    if not all_results:
        print("No results were generated. Please check your dataset paths and model configurations.")
        return None
    
    # Chuyển đổi kết quả thành DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Làm sạch DataFrame để lưu
    save_cols = [col for col in results_df.columns if col not in ['all_selected_features', 'frequent_features', 'params']]
    save_df = results_df[save_cols].copy()
    
    # Lưu kết quả
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_path = os.path.join(output_dir, f'experiment_results_{timestamp}.csv')
    save_df.to_csv(results_path, index=False)
    
    print(f"Results saved to {results_path}")
    return results_df
if __name__ == '__main__':
    # Ví dụ cách sử dụng để tìm tham số C tốt nhất cho L1SVM
    models_config = [
        {
            'model_class': L1SVM,
            'param_grid': {
                'C': [2**i for i in range(-3, 6)]  # C từ 2^-3 đến 2^5
            }
        }
    ]
    
    
    data_config = [
        {
            'dataset_name': 'diabetes',
            'dataset_types': ['original', 'noise', 'outlier', 'both']
        }
    ]
    
    results = run_full_experiment(models_config, data_config, output_dir='results')