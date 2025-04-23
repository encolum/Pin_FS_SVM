# Optimal Robust Feature Selection For Support Vector Machines With Pinball Loss

This repository contains the implementation of robust feature selection methods for Support Vector Machines using Pinball loss function. The code provides implementation for several models:

- **Pin-FS-SVM**: Pinball Loss Feature Selection SVM
- **MILP1**: Mixed Integer Linear Programming SVM
- **Pinball SVM**: Standard SVM with Pinball Loss
- **L1-SVM**: L1-norm SVM (sparse)
- **L2-SVM**: Standard SVM with L2 regularization
- **Fisher-SVM**: Fisher score-based feature selection with SVM
- **RFE-SVM**: Recursive Feature Elimination with SVM

## Project Structure

Keywords: Data Science, Outlier Detection, Feature Selection, Support Vector Machine, Mixed-Integer Linear Programming.

Support Vector Machines (SVMs) have been widely
used for classification tasks, but face significant challenges in-
cluding sensitivity to noise and outliers, and the critical need
for effective feature selection. This paper proposes a robust
Mixed Integer Linear Programming (MILP) model based on
Support Vector Machines (SVMs) that simultaneously addresses
both issues. Our model, Pin-FS-SVM, incorporates a budget
constraint to limit feature selection and employs the pinball
loss function for noise and outlier handling in the classification
process. Experiments on multiple datasets show that Pin-FS-SVM
outperforms existing methods in robustness to noise and outliers,
achieves better feature reduction, and maintains competitive
performance.
