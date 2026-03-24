"""
04_eigenvalues/eigenvalues_pca.py
==================================
Eigenvalues & Eigenvectors — geometry of transformation + PCA from scratch.

Covers:
- Computing eigenvalues / eigenvectors (numpy)
- Verifying Av = λv
- Covariance matrix eigendecomposition
- PCA from scratch (no sklearn)
- Comparing scratch PCA vs sklearn PCA
- Explained variance ratio
"""

import numpy as np

# ─────────────────────────────────────────
# 1. Eigenvalues & eigenvectors of a matrix
# ─────────────────────────────────────────
print("=== 1. Eigendecomposition ===")

A = np.array([[3.0, 1.0],
              [1.0, 3.0]])

eigenvalues, eigenvectors = np.linalg.eig(A)

print(f"Matrix A:\n{A}")
print(f"\nEigenvalues  λ: {eigenvalues}")
print(f"Eigenvectors V (columns):\n{eigenvectors}")

# ─────────────────────────────────────────
# 2. Verify Av = λv for each pair
# ─────────────────────────────────────────
print("\n=== 2. Verification: A·v = λ·v ===")
for i in range(len(eigenvalues)):
    lam = eigenvalues[i]
    v   = eigenvectors[:, i]
    Av  = A @ v
    lv  = lam * v
    match = np.allclose(Av, lv)
    print(f"  λ={lam:.2f}: A·v={Av.round(4)}  λ·v={lv.round(4)}  match={match}")

# ─────────────────────────────────────────
# 3. PCA from scratch
# ─────────────────────────────────────────
print("\n=== 3. PCA from Scratch ===")

np.random.seed(42)
n_samples = 200

# Correlated 2D data (so PCA has something meaningful to find)
mean  = [2.0, 3.0]
cov   = [[2.0, 1.5],
         [1.5, 1.2]]
X = np.random.multivariate_normal(mean, cov, n_samples)  # (200, 2)

print(f"Data shape: {X.shape}")
print(f"Original mean: {X.mean(axis=0).round(4)}")

# Step 1: Center the data
X_centered = X - X.mean(axis=0)

# Step 2: Compute covariance matrix
cov_matrix = np.cov(X_centered.T)          # (2, 2)
print(f"\nCovariance matrix:\n{cov_matrix.round(4)}")

# Step 3: Eigendecomposition of covariance matrix
vals, vecs = np.linalg.eigh(cov_matrix)    # eigh for symmetric matrices

# Step 4: Sort by descending eigenvalue
order = np.argsort(vals)[::-1]
vals  = vals[order]
vecs  = vecs[:, order]

print(f"\nEigenvalues (variance explained): {vals.round(4)}")
print(f"Principal components (columns):\n{vecs.round(4)}")

# Step 5: Project onto top-k components
k = 1   # reduce to 1D
components = vecs[:, :k]                  # (2, 1)
X_pca = X_centered @ components           # (200, 1)

print(f"\nProjected data shape: {X_pca.shape}  (200 samples → 1D)")

# Explained variance ratio
explained = vals / vals.sum()
print(f"\nExplained variance ratio: {explained.round(4)}")
print(f"  PC1 explains {explained[0]*100:.1f}% of variance")
print(f"  PC2 explains {explained[1]*100:.1f}% of variance")

# ─────────────────────────────────────────
# 4. Validate against sklearn PCA
# ─────────────────────────────────────────
print("\n=== 4. Validation vs sklearn PCA ===")
try:
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    pca.fit(X)

    print(f"sklearn explained_variance_ratio_: {pca.explained_variance_ratio_.round(4)}")
    print(f"Scratch explained variance ratio : {explained.round(4)}")
    print(f"Match: {np.allclose(explained, pca.explained_variance_ratio_, atol=1e-3)}")
except ImportError:
    print("sklearn not installed — skipping validation (pip install scikit-learn)")

# ─────────────────────────────────────────
# 5. High-dimensional PCA: 100D → 2D
# ─────────────────────────────────────────
print("\n=== 5. High-Dimensional PCA: 100D → 2D ===")

np.random.seed(7)
X_high = np.random.randn(500, 100)   # 500 samples, 100 features

X_c    = X_high - X_high.mean(axis=0)
cov_h  = np.cov(X_c.T)
vals_h, vecs_h = np.linalg.eigh(cov_h)
order_h = np.argsort(vals_h)[::-1]
vals_h  = vals_h[order_h]
vecs_h  = vecs_h[:, order_h]

X_2d = X_c @ vecs_h[:, :2]

explained_h = vals_h[:2].sum() / vals_h.sum()
print(f"Original shape  : {X_high.shape}")
print(f"Reduced shape   : {X_2d.shape}")
print(f"Variance kept   : {explained_h*100:.1f}%  (random data → low, expected)")
print("→ PCA = eigendecomposition of the covariance matrix. No magic.")
