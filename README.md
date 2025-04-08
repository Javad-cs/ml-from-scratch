# Machine Learning From Scratch 

This repository contains a collection of foundational machine learning projects implemented entirely from scratch using NumPy, PyTorch, and other low-level tools, without relying on high-level machine learning APIs. Each project is an independent notebook focused on understanding how core ML algorithms work under the hood.

These implementations were built as part of a hands-on effort to deeply understand key concepts in supervised, unsupervised, and reinforcement learning.

---

## Projects

###  `lasso_ridge_reg/`
**Lasso & Ridge Regression**  
Manual implementation of linear regression with L1 (Lasso) and L2 (Ridge) regularization using gradient descent. Includes custom cost functions, gradient logic, and weight updates.

- Implements regularization from scratch
- Visualizes weight shrinkage behavior
- Comparison with scikit-learn (optional)

---

### `Kmeans/`
**K-Means Clustering**  
An unsupervised clustering algorithm implemented manually using NumPy — no external clustering libraries used.

- Random centroid initialization
- Vectorized distance computation
- Iterative clustering until convergence

---

### `GMM_EM/`
**Gaussian Mixture Models with Expectation-Maximization**  
A from-scratch implementation of the EM algorithm to train GMMs on a multivariate dataset.

- Uses `scipy.stats` for multivariate Gaussians
- Updates responsibilities, means, covariances iteratively
- Log-likelihood computation for convergence check

---

### `SVM-KSVM/`
**Support Vector Machine & Kernel SVM**  
Manual implementation of linear SVM with hinge loss, and kernelized SVM using `cvxopt` for solving the quadratic programming problem.

- Gradient-based linear SVM
- QP-based kernel SVM (polynomial, RBF kernels)
- Dataset split across training, validation, test sets

---

### `RL/`
**Reinforcement Learning in a Custom Environment**  
Implements a basic reinforcement learning agent (epsilon-greedy strategy) in a custom visual environment built with `pygame`.

- Custom gridworld with start, goal, ice, rocks, and holes
- Q-table logic for action selection and learning
- Visualization using `pygame` + `imageio` for GIF exports

---

##  Goal of This Repo

This repository is not meant to be a production-grade ML toolkit. Instead, it’s a learning lab, a place to:

- Understand algorithms by implementing them
- Experiment with data, math, and code
- Move beyond "import and go" ML workflows

---

##  Tech Stack

- Python (NumPy, Matplotlib, PIL, PyTorch)
- `cvxopt` (for QP in SVM)
- `pygame` (for custom RL environment)
- `scikit-learn` only for optional benchmarking

---

## Folder Structure

Each folder is a standalone project containing:

-  A Jupyter Notebook with full code and explanations  
-  `.npy` datasets (for projects that require custom or preprocessed data)  
-  Image assets (for RL environment visuals)  

```text
├── lasso_ridge_reg/
│   └── lasso_ridge_reg.ipynb
│   ↳ Implements linear regression with L1 & L2 regularization from scratch

├── kmeans/
│   └── kmeans.ipynb
│   ↳ Custom K-Means clustering with random init & iterative updates

├── GMM_EM/
│   ├── GMM_EM.ipynb
│   └── Dataset(X_data).npy
│   ↳ Gaussian Mixture Model trained using the EM algorithm

├── SVM-KSVM/
│   ├── SVM_KSVM.ipynb
│   ├── Dataset0(X_data).npy
│   ├── Dataset1(Train).npy / (Validation).npy / (Test).npy
│   └── Dataset2(...)
│   ↳ SVM & Kernel SVM using gradient descent and CVXOPT QP

├── RL/
│   ├── RL.ipynb
│   └── img/
│       └── (visual assets like elf, goal, rock, etc.)
│   ↳ Reinforcement learning in a custom Pygame gridworld
```