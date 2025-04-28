# src/models/rfe_svm.py

import numpy as np
import time
from docplex.mp.model import Model

class RFESVM:
    """
    Standard Recursive Feature Elimination with SVM baseline
    """
    def __init__(self, n_features, C=1.0, kernel='linear', time_limit=None):
        """
        Parameters:
        -----------
        n_features : int
            Target number of features to select.
        C : float
            Regularization parameter for SVM.
        kernel : str
            Kernel type (only 'linear' supported).
        time_limit : int or None
            Time limit for QP solver in seconds.
        """
        if n_features is None or n_features < 1:
            raise ValueError("n_features must be a positive integer")
        self.n_features = n_features
        self.C = C
        self.kernel = kernel
        self.time_limit = time_limit
        self.selected_indices = None
        self.w = None
        self.b = None
        self.train_time = None

    def _svm_train_qp(self, X, y):
        """
        Train a linear SVM via QP with L1 regularization
        Returns (w_opt, b_opt)
        """
        opt_mod = Model(name='L1-SVM')
        if self.time_limit:
            opt_mod.set_time_limit(self.time_limit)
        m, n = X.shape

        # decision variables
        w = opt_mod.continuous_var_list(n, name='w')
        b = opt_mod.continuous_var(name='b')
        v = opt_mod.continuous_var_list(n, name='v', lb=0)
        xi = opt_mod.continuous_var_list(m, name='xi', lb=0)

        # objective
        opt_mod.minimize(opt_mod.sum(v[j] for j in range(n))
                         + self.C * opt_mod.sum(xi[i] for i in range(m)))

        # constraints
        for i in range(m):
            opt_mod.add_constraint(
                y[i] * (opt_mod.sum(w[j] * X[i,j] for j in range(n)) + b)
                >= 1 - xi[i]
            )
        for j in range(n):
            opt_mod.add_constraint(w[j] <= v[j])
            opt_mod.add_constraint(-w[j] <= v[j])

        sol = opt_mod.solve()
        if sol is None:
            return None, None
        w_opt = np.array([sol.get_value(w[j]) for j in range(n)])
        b_opt = sol.get_value(b)
        return w_opt, b_opt

    def fit(self, X, y):
        """
        Fit the RFE-SVM model, eliminating one feature per iteration until n_features remain.
        """
        start = time.time()
        n_samples, n_total = X.shape
        n_select = min(self.n_features, n_total)
        remaining = list(range(n_total))

        # recursive elimination: remove one feature at a time
        while len(remaining) > n_select:
            X_sub = X[:, remaining]
            w_sub, b_sub = self._svm_train_qp(X_sub, y)
            if w_sub is None:
                break
            # importance by squared weights
            importance = w_sub**2
            # index within remaining to drop
            drop_idx = int(np.argmin(importance))
            del remaining[drop_idx]

        # final training on selected features
        self.selected_indices = remaining
        X_final = X[:, self.selected_indices]
        w_fin, b_fin = self._svm_train_qp(X_final, y)
        self.w = np.zeros(n_total)
        if w_fin is not None:
            for i, idx in enumerate(self.selected_indices):
                self.w[idx] = w_fin[i]
            self.b = b_fin
        else:
            self.b = 0.0
        self.train_time = time.time() - start
        return self

    def predict(self, X):
        if self.w is None:
            raise ValueError("Model not fitted yet")
        X_sel = X[:, self.selected_indices]
        scores = np.dot(X_sel, self.w[self.selected_indices]) + self.b
        return np.sign(scores)

    def get_selected_features(self):
        if self.selected_indices is None:
            raise ValueError("Model not fitted yet")
        # return 1-based indices
        return [i + 1 for i in self.selected_indices]

    def get_num_selected_features(self):
        return len(self.get_selected_features())
