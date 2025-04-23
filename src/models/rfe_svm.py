import numpy as np
import time
from sklearn.svm import SVC
from sklearn.feature_selection import RFE

class RFESVM:
    """
    Recursive Feature Elimination with SVM
    """
    
    def __init__(self, n_features=None, C=1.0, kernel='linear', time_limit=None, step=1):
        """
        Initialize RFE + SVM model
        
        Parameters:
        -----------
        n_features : int or None
            Number of features to select. If None, use all features.
        C : float
            Regularization parameter for SVM
        kernel : str
            Kernel type for SVM ('linear', 'rbf', etc.)
        time_limit : int or None
            Time limit (not used in this implementation)
        step : int
            Number of features to remove at each iteration
        """
        self.n_features = n_features
        self.C = C
        self.kernel = kernel
        self.time_limit = time_limit
        self.step = step
        self.selector = None
        self.model = None
        self.w = None
        self.b = None
        self.train_time = None
    
    def fit(self, X, y):
        """
        Fit the RFE + SVM model
        
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
        
        n_samples, n_features = X.shape
        n_select = self.n_features if self.n_features is not None else n_features
        n_select = min(n_select, n_features)
        
        # Create SVM estimator
        estimator = SVC(C=self.C, kernel=self.kernel)
        
        # Create RFE selector
        self.selector = RFE(
            estimator=estimator,
            n_features_to_select=n_select,
            step=self.step
        )
        
        # Fit RFE
        self.selector.fit(X, y)
        
        # Get selected feature mask
        feature_mask = self.selector.support_
        
        # Create reduced feature matrix
        X_selected = X[:, feature_mask]
        
        # Train final SVM on selected features
        self.model = SVC(C=self.C, kernel=self.kernel)
        self.model.fit(X_selected, y)
        
        # Extract weights for linear kernel
        if self.kernel == 'linear':
            w_selected = self.model.coef_[0]
            self.w = np.zeros(n_features)
            for i, selected in enumerate(feature_mask):
                if selected:
                    idx = np.where(feature_mask[:i+1])[0].size - 1
                    self.w[i] = w_selected[idx]
            self.b = self.model.intercept_[0]
        else:
            self.w = np.zeros(n_features)
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
        if self.selector is None or self.model is None:
            raise ValueError("Model not fitted yet")
        
        X_selected = X[:, self.selector.support_]
        return self.model.predict(X_selected)
    
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
        if self.selector is None or self.model is None:
            raise ValueError("Model not fitted yet")
        
        X_selected = X[:, self.selector.support_]
        return self.model.decision_function(X_selected)
    
    def get_selected_features(self):
        """
        Get indices of selected features
        
        Returns:
        --------
        list
            Indices of selected features (1-indexed)
        """
        if self.selector is None:
            raise ValueError("Model not fitted yet")
        
        selected = np.where(self.selector.support_)[0]
        return [idx + 1 for idx in selected]
    
    def get_num_selected_features(self):
        """
        Get the number of selected features
        
        Returns:
        --------
        int
            Number of selected features
        """
        return len(self.get_selected_features())
