import numpy as np
from docplex.mp.model import Model
import time

class FisherSVM:
    """
    Fisher Score Feature Selection with SVM
    """
    
    def __init__(self, n_features=None, C=1.0, kernel='linear', time_limit=None):
        """
        Initialize Fisher Score + SVM model
        
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
        """
        self.n_features = n_features
        self.C = C
        self.kernel = kernel
        self.time_limit = time_limit
        self.selected_indices = None
        self.model = None
        self.w = None
        self.b = None
        self.train_time = None
    
    def _calculate_f_score(self, X, y):
        """
        Calculate Fisher score for feature selection
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix
        y : numpy array
            Target labels (-1/1)
            
        Returns:
        --------
        f_scores : numpy array
            F-score for each feature
        """
        pos_indices = y == 1
        neg_indices = y == -1
        
        X_positive = X[pos_indices]
        X_negative = X[neg_indices]
        
        n_positive = X_positive.shape[0]
        n_negative = X_negative.shape[0]
        
        pos_mean = np.mean(X_positive, axis=0)
        neg_mean = np.mean(X_negative, axis=0)
        total_mean = np.mean(X, axis=0)
        
        # Handle division by zero when calculating variance
        variance_positive = np.sum((X_positive - pos_mean)**2, axis=0) / (n_positive - 1) if n_positive > 1 else 1e-10
        variance_negative = np.sum((X_negative - neg_mean)**2, axis=0) / (n_negative - 1) if n_negative > 1 else 1e-10
        
        numerator = (pos_mean - total_mean)**2 + (neg_mean - total_mean)**2
        denominator = variance_positive + variance_negative
        
        # Avoid division by zero
        denominator = np.where(denominator == 0, np.inf, denominator)
        f_scores = numerator / denominator
        
        # Replace NaN and inf values with 0
        f_scores = np.nan_to_num(f_scores, nan=0.0, posinf=0.0, neginf=0.0)
        
        return f_scores
    
    
    def _svm_train_qp(self, X, y):
        """
        Train SVM using quadratic programming with L1 regularization
        
        Parameters:
        -----------
        X : numpy array
            Training feature matrix
        y : numpy array
            Training labels (-1/1)
            
        Returns:
        --------
        tuple
            (w_opt, b_opt) - optimal weights and bias
        """
        # Initialize the model
        opt_mod = Model(name='L1-SVM')
        if self.time_limit:
            opt_mod.set_time_limit(self.time_limit)
            
        # Number of samples and features
        m, n = X.shape

        # Decision variables
        w = opt_mod.continuous_var_list(n, name='w')
        b = opt_mod.continuous_var(name='b')
        v = opt_mod.continuous_var_list(n, name='v', lb=0)
        xi = opt_mod.continuous_var_list(m, lb=0, name='xi')

        # Objective function
        opt_mod.minimize(opt_mod.sum(v[j] for j in range(n)) + self.C * opt_mod.sum(xi[i] for i in range(m)))

        # Constraints
        for i in range(m):
            opt_mod.add_constraint(y[i] * (opt_mod.sum(w[j] * X[i, j] for j in range(n)) + b) >= 1 - xi[i])
        for j in range(n):
            opt_mod.add_constraint(w[j] <= v[j])
            opt_mod.add_constraint(-v[j] <= w[j])
        
        solution = opt_mod.solve()
        if solution:
            w_opt = np.array([solution.get_value(w[j]) for j in range(n)])
            b_opt = solution.get_value(b)
            return w_opt, b_opt
        return None, None
    
    def fit(self, X, y):
        """
        Fit the Fisher Score + SVM model
        
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
        
        # Calculate Fisher scores
        f_scores = self._calculate_f_score(X, y)
        
        # Select top features
        self.selected_indices = np.argsort(f_scores)[::-1][:n_select]
        
        # Create reduced feature matrix
        X_selected = X[:, self.selected_indices]
        
        # Train SVM on selected features using QP solver
        w_selected, b = self._svm_train_qp(X_selected, y)
        
        if w_selected is not None:
            # Map selected feature weights back to original feature space
            self.w = np.zeros(n_features)
            for i, idx in enumerate(self.selected_indices):
                self.w[idx] = w_selected[i]
            self.b = b
        else:
            # Handle case where QP solver fails
            self.w = np.zeros(n_features)
            self.b = 0.0
            print("Warning: QP optimization failed - using zero weights")
        
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
        if self.w is None:
            raise ValueError("Model not fitted yet")
        
        # Use selected features to make predictions
        X_selected = X[:, self.selected_indices]
        scores = np.dot(X_selected, self.w[self.selected_indices]) + self.b
        return np.sign(scores)
    
    def get_selected_features(self):
        """
        Get indices of selected features
        
        Returns:
        --------
        list
            Indices of selected features (1-indexed)
        """
        if self.selected_indices is None:
            raise ValueError("Model not fitted yet")
        
        return [idx + 1 for idx in self.selected_indices]
    
    def get_num_selected_features(self):
        """
        Get the number of selected features
        
        Returns:
        --------
        int
            Number of selected features
        """
        return len(self.get_selected_features())
