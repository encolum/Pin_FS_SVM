import os
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

# Import các module cần thiết từ dự án
from src.models.l2_svm import L2SVM
from src.utils.metrics import evaluate_model, count_selected_features
# Fix import error - use the correct function name
from src.experiment.run_experiment import run_single_experiment_with_param_tuning

def test_model_directly():
    """Test L2SVM trực tiếp trên dữ liệu breast cancer từ scikit-learn"""
    print("====== TEST 1: KIỂM TRA TRỰC TIẾP MÔ HÌNH L2SVM ======")
    # Load dữ liệu
    data = load_breast_cancer()
    X = data.data
    y = 2 * data.target - 1  # Chuyển 0/1 thành -1/1
    
    # Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Khởi tạo và huấn luyện mô hình
    print("Huấn luyện L2SVM...")
    start_time = time.time()
    model = L2SVM(C=1.0)
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Dự đoán
    y_pred = model.predict(X_test)
    y_score = model.decision_function(X_test)
    
    # Đánh giá
    metrics = evaluate_model(y_test, y_pred, y_score)
    num_features, selected_features = count_selected_features(model.w)
    
    print(f"Thời gian huấn luyện: {train_time:.4f}s")
    print(f"Số đặc trưng được chọn: {num_features} / {X.shape[1]}")
    print("Kết quả đánh giá:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

def test_run_single_experiment():
    """Test hàm run_single_experiment_with_param_tuning với L2SVM"""
    print("\n====== TEST 2: KIỂM TRA HÀM RUN_SINGLE_EXPERIMENT_WITH_PARAM_TUNING ======")
    # Thực hiện một thí nghiệm đơn lẻ
    result = run_single_experiment_with_param_tuning(
        model_class=L2SVM,
        param_grid={'C': [2**i for i in range(-3, 6)]},  # Use a param_grid instead of model_params
        dataset_name='wdbc',  # Sử dụng tập dữ liệu WDBC có sẵn
        dataset_type='original',
        n_splits=10  # Chỉ chạy 3 fold để tiết kiệm thời gian
    )
    
    if result:
        print("\nKết quả thí nghiệm:")
        print(f"  Dataset: {result['dataset']} ({result['dataset_type']})")
        print(f"  Model: {result['model']}")
        print(f"  Accuracy: {result['accuracy_mean']:.4f} ± {result['accuracy_std']:.4f}")
        print(f"  AUC: {result['auc_mean']:.4f} ± {result['auc_std']:.4f}")
        if 'f1_score_mean' in result:
            print(f"  F1-score: {result['f1_score_mean']:.4f} ± {result['f1_score_std']:.4f}")
        if 'g_mean_mean' in result:
            print(f"  G-mean: {result['g_mean_mean']:.4f} ± {result['g_mean_std']:.4f}")
        print(f"  Số đặc trưng được chọn: {result['num_features_mean']:.1f} ± {result['num_features_std']:.1f}")
        print(f"  Thời gian huấn luyện: {result['train_time_mean']:.4f}s ± {result['train_time_std']:.4f}s")
    else:
        print("Không có kết quả từ thí nghiệm!")

if __name__ == "__main__":
    # Chạy các bài test
    # try:
    #     test_model_directly()
    # except Exception as e:
    #     print(f"Lỗi khi chạy test_model_directly: {e}")
    
    try:
        test_run_single_experiment()
    except Exception as e:
        print(f"Lỗi khi chạy test_run_single_experiment: {e}")