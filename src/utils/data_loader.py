import os
import pandas as pd
import numpy as np

def load_dataset(dataset_name, dataset_type="original"):
    """
    Load a dataset with specified type (original, noise, outlier, or both)
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset ('wdbc', 'diabetes', 'cleveland', 'ionosphere', 'sonar', 'australia')
    dataset_type : str
        Type of dataset ('original', 'noise', 'outlier', 'both')
        
    Returns:
    --------
    X : numpy array
        Feature matrix
    y : numpy array
        Target labels (-1/1)
    """
    
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'Dataset')
    
    dataset_paths = {
        'wdbc': {
            'original': os.path.join(base_dir, 'wdbc', 'wdbc.data.txt'),
            'noise': os.path.join(base_dir, 'wdbc', 'wdbc_noisy_label_feature.txt'),
            'outlier': os.path.join(base_dir, 'wdbc', 'wdbc_noisy_label_outlier.txt'),
            'both': os.path.join(base_dir, 'wdbc', 'wdbc_both_noise_outlier.txt')
        },
        'diabetes': {
            'original': os.path.join(base_dir, 'diabetes', 'diabetes.csv'),
            'noise': os.path.join(base_dir, 'diabetes', 'diabetes_noise_label_feature.csv'),
            'outlier': os.path.join(base_dir, 'diabetes', 'diabetes_outlier.csv'),
            'both': os.path.join(base_dir, 'diabetes', 'diabetes_both_noise_outlier.csv')
        },
        'cleveland': {
            'original': os.path.join(base_dir, 'cleveland', 'Heart_disease_cleveland_new.csv'),
            'noise': os.path.join(base_dir, 'cleveland', 'clevaland_noise_label_feature.csv'),
            'outlier': os.path.join(base_dir, 'cleveland', 'clevaland_outlier.csv'),
            'both': os.path.join(base_dir, 'cleveland', 'cleveland_both_noise_outlier.csv')
        },
        'ionosphere': {
            'original': os.path.join(base_dir, 'ionosphere', 'ionosphere.data'),
            'noise': os.path.join(base_dir, 'ionosphere', 'ionosphere_noise_label_feature.txt'),
            'outlier': os.path.join(base_dir, 'ionosphere', 'ionosphere_outlier.txt'),
            'both': os.path.join(base_dir, 'ionosphere', 'ionosphere_both_noise_outlier.txt')
        },
        'sonar': {
            'original': os.path.join(base_dir, 'sonar', 'sonar.txt'),
            'noise': os.path.join(base_dir, 'sonar', 'sonar_noise_label_feature.txt'),
            'outlier': os.path.join(base_dir, 'sonar', 'sonar_outlier.txt'),
            'both': os.path.join(base_dir, 'sonar', 'sonar_both_noise_outlier.txt')
        },
        'australia': {
            'original': os.path.join(base_dir, 'australia', 'australia.txt'),
            'noise': os.path.join(base_dir, 'australia', 'australia_noise_label_feature.txt'),
            'outlier': os.path.join(base_dir, 'australia', 'australia_outlier.txt'),
            'both': os.path.join(base_dir, 'australia', 'australia_both_noise_outlier.txt')
        }
    }
    
    try:
        file_path = dataset_paths[dataset_name][dataset_type]
        
        # Process different dataset formats
        if dataset_name == 'wdbc':
            df = pd.read_csv(file_path, header=None)
            X = df.iloc[:, 2:].values
            y = df.iloc[:, 1].values
            y = np.where(y == 'M', 1, -1)  # Convert B/M to -1/1
        
        elif dataset_name == 'diabetes':
            df = pd.read_csv(file_path)
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            y = np.where(y == 0, -1, 1)  # Convert 0/1 to -1/1
        
        elif dataset_name == 'cleveland':
            df = pd.read_csv(file_path, header=None)
            X = df.iloc[1:, 0:13].values.astype(float)
            y = df.iloc[1:, 13].values.astype(float)
            y = np.where(y == 0, -1, 1)  # Convert 0/1 to -1/1
        
        elif dataset_name == 'ionosphere':
            df = pd.read_csv(file_path, header=None)
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            y = np.where(y == 'g', 1, -1)  # Convert g/b to 1/-1
        
        elif dataset_name == 'sonar':
            df = pd.read_csv(file_path, header=None)
            X = df.iloc[:, 0:60].values
            y = df.iloc[:, 60].values
            y = np.where(y == 'M', 1, -1)  # Convert to -1/1
        
        elif dataset_name == 'australia':
            # Special handling for the original australia dataset which uses spaces as delimiter
            if dataset_type == 'original':
                df = pd.read_csv(file_path, header=None, sep=' ')
            else:
                df = pd.read_csv(file_path, header=None)
            
            X = df.iloc[:, :14].values
            y = df.iloc[:, 14].values
            y = np.where(y == 0, -1, 1)  # Convert 0/1 to -1/1
        
        return X, y
        
    except Exception as e:
        print(f"Error loading dataset {dataset_name} ({dataset_type}): {e}")
        print(f"Please ensure the Dataset folder exists at: {base_dir}")
        print(f"And the file exists at: {dataset_paths.get(dataset_name, {}).get(dataset_type, 'Unknown path')}")
        
        # Return empty data as fallback
        return np.array([]), np.array([])
