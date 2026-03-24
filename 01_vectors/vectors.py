"""
01_vectors/vectors.py
=====================
Vectors — the atoms of ML data.

Covers:
- Creating vectors
- Vector addition & scalar multiplication
- Magnitude (L2 norm)
- Unit vectors (normalization)
- Cosine similarity
- Why vectors matter in ML
"""

import numpy as np

# ─────────────────────────────────────────
# 1. Creating vectors
# ─────────────────────────────────────────
v = np.array([3.0, 1.5, -0.8])
u = np.array([1.0, 2.0,  0.5])

print("=== 1. Basic Vectors ===")
print(f"v = {v}")
print(f"u = {u}")

# ─────────────────────────────────────────
# 2. Vector addition & scalar multiplication
# ─────────────────────────────────────────
print("\n=== 2. Arithmetic ===")
print(f"v + u        = {v + u}")          # element-wise addition
print(f"v - u        = {v - u}")          # element-wise subtraction
print(f"2 * v        = {2 * v}")          # scalar multiply (stretches vector)
print(f"v * u        = {v * u}")          # element-wise (Hadamard) product

# ─────────────────────────────────────────
# 3. Magnitude (L2 norm)
# ─────────────────────────────────────────
print("\n=== 3. Magnitude (L2 Norm) ===")
magnitude_v = np.linalg.norm(v)
magnitude_u = np.linalg.norm(u)
print(f"‖v‖ = {magnitude_v:.4f}")
print(f"‖u‖ = {magnitude_u:.4f}")

# Manual calculation to show what norm does under the hood
manual_norm = np.sqrt(np.sum(v ** 2))
print(f"Manual ‖v‖ = sqrt({' + '.join([f'{x:.2f}²' for x in v])}) = {manual_norm:.4f}")

# ─────────────────────────────────────────
# 4. Unit vector (normalization)
# ─────────────────────────────────────────
print("\n=== 4. Unit Vector (Normalization) ===")
v_hat = v / np.linalg.norm(v)
print(f"v_hat = {v_hat}")
print(f"‖v_hat‖ = {np.linalg.norm(v_hat):.6f}  ← always 1.0")
print("Unit vectors carry DIRECTION only — magnitude stripped.")

# ─────────────────────────────────────────
# 5. Cosine similarity
# ─────────────────────────────────────────
print("\n=== 5. Cosine Similarity ===")

def cosine_similarity(a, b):
    """Measures angle between two vectors. Range: -1 (opposite) to +1 (identical direction)."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

sim_uv = cosine_similarity(u, v)
print(f"cosine_similarity(u, v) = {sim_uv:.4f}")

# Identical direction → similarity = 1
w = 5 * v
print(f"cosine_similarity(v, 5v) = {cosine_similarity(v, w):.4f}  ← same direction")

# Perpendicular → similarity = 0
perp = np.array([v[1], -v[0], 0])
print(f"cosine_similarity(v, perp) ≈ {cosine_similarity(v, perp):.4f}  ← perpendicular")

# ─────────────────────────────────────────
# 6. ML context — a sentence as a vector
# ─────────────────────────────────────────
print("\n=== 6. ML Context: Sentence Vectors ===")
# Simplified bag-of-words vectors (vocab: [cat, dog, sat, mat, ran])
sentence_a = np.array([1, 0, 1, 1, 0], dtype=float)   # "cat sat mat"
sentence_b = np.array([1, 0, 1, 0, 0], dtype=float)   # "cat sat"
sentence_c = np.array([0, 1, 0, 0, 1], dtype=float)   # "dog ran"

print(f"'cat sat mat' vs 'cat sat'  → similarity = {cosine_similarity(sentence_a, sentence_b):.4f}")
print(f"'cat sat mat' vs 'dog ran'  → similarity = {cosine_similarity(sentence_a, sentence_c):.4f}")
print("→ Cosine similarity correctly finds semantically closer sentences.")
