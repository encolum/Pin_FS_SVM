from sklearn.datasets import fetch_openml
import numpy as np

# GINA
X, y = fetch_openml(data_id=1038, as_frame=False, return_X_y=True)
np.savez("gina.npz", X=X, y=y)

# HIVA
X, y = fetch_openml(data_id=1039, as_frame=False, return_X_y=True)
np.savez("hiva.npz", X=X, y=y)