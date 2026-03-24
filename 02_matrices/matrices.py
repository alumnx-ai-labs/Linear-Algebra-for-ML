"""
02_matrices/matrices.py
=======================
Matrices — grids of transformation.

Covers:
- Matrix creation & shapes
- Matrix multiplication (@ operator)
- Transpose
- Element-wise vs matrix multiply
- A neural network dense layer from scratch
- Broadcasting
"""

import numpy as np

# ─────────────────────────────────────────
# 1. Creating matrices & checking shapes
# ─────────────────────────────────────────
print("=== 1. Matrix Basics ===")
A = np.array([[1, 2],
              [3, 4],
              [5, 6]])   # shape (3, 2)

B = np.array([[7,  8,  9],
              [10, 11, 12]])  # shape (2, 3)

print(f"A shape: {A.shape}\n{A}\n")
print(f"B shape: {B.shape}\n{B}\n")

# ─────────────────────────────────────────
# 2. Matrix multiplication  (m×k) @ (k×n) → (m×n)
# ─────────────────────────────────────────
print("=== 2. Matrix Multiply (A @ B) ===")
C = A @ B    # (3,2) @ (2,3) → (3,3)
print(f"C = A @ B, shape: {C.shape}\n{C}\n")

# Manual to show what happens: C[i,j] = dot(A[i,:], B[:,j])
print("Manual C[0,0] = dot(A[0,:], B[:,0]) =", np.dot(A[0, :], B[:, 0]))

# ─────────────────────────────────────────
# 3. Transpose
# ─────────────────────────────────────────
print("\n=== 3. Transpose ===")
print(f"A.T shape: {A.T.shape}\n{A.T}\n")

# ─────────────────────────────────────────
# 4. Element-wise vs matrix multiply
# ─────────────────────────────────────────
print("=== 4. Element-wise (*) vs Matrix Multiply (@) ===")
X = np.array([[1, 2], [3, 4]])
Y = np.array([[5, 6], [7, 8]])

print(f"X * Y  (element-wise / Hadamard):\n{X * Y}\n")
print(f"X @ Y  (matrix multiply):\n{X @ Y}\n")
print("⚠  These are NOT the same. Confusing them is the #1 bug in ML code.")

# ─────────────────────────────────────────
# 5. Dense neural network layer from scratch
# ─────────────────────────────────────────
print("=== 5. Dense Layer: y = activation(W @ x + b) ===")

np.random.seed(42)

input_dim  = 4
output_dim = 3
batch_size = 5

# Weight matrix and bias (randomly initialized)
W = np.random.randn(output_dim, input_dim) * 0.1   # shape (3, 4)
b = np.zeros(output_dim)                            # shape (3,)

# Input batch: 5 samples, each 4-dimensional
X_batch = np.random.randn(batch_size, input_dim)   # shape (5, 4)

# Forward pass: one matrix multiply
logits = X_batch @ W.T + b    # (5,4) @ (4,3) + (3,) → (5,3)

# ReLU activation
relu = lambda z: np.maximum(0, z)
output = relu(logits)

print(f"W shape: {W.shape}")
print(f"X_batch shape: {X_batch.shape}")
print(f"logits shape: {logits.shape}")
print(f"output (after ReLU) shape: {output.shape}")
print(f"\nFirst sample output: {output[0]}")
print("→ One matrix multiply transforms a whole batch simultaneously.")

# ─────────────────────────────────────────
# 6. Broadcasting
# ─────────────────────────────────────────
print("\n=== 6. Broadcasting ===")
M = np.array([[1, 2, 3],
              [4, 5, 6]])   # shape (2, 3)

row_bias = np.array([10, 20, 30])   # shape (3,) — broadcasts across rows

print(f"M:\n{M}")
print(f"row_bias: {row_bias}")
print(f"M + row_bias:\n{M + row_bias}")
print("→ Broadcasting adds the bias to every row without explicit loops.")
