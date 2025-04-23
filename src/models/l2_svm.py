import numpy as np
import time
from sklearn.svm import SVC

class L2SVM:
    """
    Standard L2-regularized Support Vector Machine
    Wrapper around sklearn's SVC
    """
    
    def __init__(self, C=1.0, kernel='linear', time_limit=None):
        """
        Initialize L2-SVM model
        
        Parameters:
        -----------
        C : float
            Regularization parameter
        kernel : str
            Kernel type ('linear', 'rbf', 'poly', etc.)
        time_limit : int or None
            Time limit for optimization (not used by sklearn)
        """
        self.C = C
        self.kernel = kernel
        self.time_limit = time_limit
        self.model = None
        self.w = None
        self.b = None
        self.train_time = None
    
    def fit(self, X, y):
        """
        Fit the L2-SVM model
        
        Parameters:
        -----------
        X : numpy array
            Training feature matrix
        y : numpy array
            Training labels (-1/1)
            
        Returns:
        --------
        self
        """
        start_time = time.time()
        
        # Initialize and fit SVM
        self.model = SVC(C=self.C, kernel=self.kernel, probability=True)
        self.model.fit(X, y)
        
        # For linear kernel, extract weights and bias
        if self.kernel == 'linear':
            self.w = self.model.coef_[0]
            self.b = self.model.intercept_[0]
        else:
            # For non-linear kernels, we don't have explicit weights
            self.w = np.zeros(X.shape[1])
            self.b = self.model.intercept_[0]
        
        self.train_time = time.time() - start_time
        return self
    
    def predict(self, X):
        """
        Make predictions using the fitted model
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix
        
        Returns:
        --------
        numpy array
            Predicted labels (-1/1)
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        return self.model.predict(X)
    
    def decision_function(self, X):
        """
        Calculate decision function scores
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix
        
        Returns:
        --------
        numpy array
            Decision function scores
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        return self.model.decision_function(X)
    
    def get_selected_features(self):
        """
        Get indices of selected features
        For L2-SVM all features are typically used
        
        Returns:
        --------
        list
            Indices of selected features (1-indexed)
        """
        if self.w is None:
            raise ValueError("Model not fitted yet")
        
        if self.kernel == 'linear':
            # For linear kernel, features with non-zero coefficients
            return [j + 1 for j in range(len(self.w)) if abs(self.w[j]) > 1e-6]
        else:
            # For non-linear kernels, we don't have explicit feature selection
            return list(range(1, len(self.w) + 1))
    
    def get_num_selected_features(self):
        """
        Get the number of selected features
        
        Returns:
        --------
        int
            Number of selected features
        """
        return len(self.get_selected_features())
