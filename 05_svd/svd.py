"""
05_svd/svd.py
=============
SVD — Singular Value Decomposition: the crown jewel of linear algebra.

Covers:
- SVD decomposition and shape inspection
- Low-rank approximation (image compression)
- Explained variance via singular values
- Truncated SVD for noise reduction (LSA-style)
- Collaborative filtering / recommender systems via SVD
"""

import numpy as np

# ─────────────────────────────────────────
# 1. SVD decomposition — A = U Σ Vᵀ
# ─────────────────────────────────────────
print("=== 1. SVD Decomposition ===")

A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10,11,12]], dtype=float)  # shape (4, 3)

U, S, Vt = np.linalg.svd(A, full_matrices=False)

print(f"A shape : {A.shape}")
print(f"U shape : {U.shape}   (left singular vectors  — 'output space')")
print(f"S shape : {S.shape}   (singular values         — importance ranking)")
print(f"Vt shape: {Vt.shape}  (right singular vectors  — 'input space')")
print(f"\nSingular values: {S.round(4)}")

# Reconstruct A from U, S, Vt
A_reconstructed = U @ np.diag(S) @ Vt
print(f"\nReconstruction error: {np.linalg.norm(A - A_reconstructed):.2e}  (should be ~0)")

# ─────────────────────────────────────────
# 2. Low-rank approximation (image compression analogy)
# ─────────────────────────────────────────
print("\n=== 2. Low-Rank Approximation (Image Compression Analogy) ===")

np.random.seed(42)
# Simulate a 20×20 grayscale image patch
image = np.random.randint(50, 200, size=(20, 20)).astype(float)

U_img, S_img, Vt_img = np.linalg.svd(image, full_matrices=False)

total_energy = np.sum(S_img ** 2)

for k in [1, 3, 5, 10, 20]:
    approx = U_img[:, :k] @ np.diag(S_img[:k]) @ Vt_img[:k, :]
    error  = np.linalg.norm(image - approx) / np.linalg.norm(image)
    energy = np.sum(S_img[:k] ** 2) / total_energy
    params_original = 20 * 20
    params_compressed = k * (20 + 20 + 1)
    print(f"  k={k:2d} | energy kept={energy*100:5.1f}% | rel error={error:.4f} | "
          f"params: {params_original} → {params_compressed} (ratio {params_original/params_compressed:.1f}x)")

# ─────────────────────────────────────────
# 3. Explained variance via singular values
# ─────────────────────────────────────────
print("\n=== 3. Explained Variance via Singular Values ===")

np.random.seed(1)
data = np.random.randn(100, 10)
data[:, 0] *= 5    # make first feature dominant
data[:, 1] *= 3

U_d, S_d, Vt_d = np.linalg.svd(data, full_matrices=False)
variance_explained = S_d**2 / np.sum(S_d**2)
cumulative = np.cumsum(variance_explained)

print("Component | Variance Explained | Cumulative")
for i, (v, c) in enumerate(zip(variance_explained[:6], cumulative[:6])):
    print(f"  PC {i+1:2d}   |     {v*100:5.1f}%          |   {c*100:5.1f}%")

# ─────────────────────────────────────────
# 4. LSA — Latent Semantic Analysis (noise reduction)
# ─────────────────────────────────────────
print("\n=== 4. LSA / Truncated SVD (Document-Term Matrix) ===")

# Tiny document-term matrix (rows=docs, cols=words)
# Words: [cat, dog, feline, canine, python, java]
doc_term = np.array([
    [1, 0, 1, 0, 0, 0],   # "cat feline"
    [0, 1, 0, 1, 0, 0],   # "dog canine"
    [1, 0, 0, 0, 0, 0],   # "cat"
    [0, 0, 0, 0, 1, 1],   # "python java"
    [0, 0, 0, 0, 1, 0],   # "python"
], dtype=float)

words = ["cat", "dog", "feline", "canine", "python", "java"]

U_lsa, S_lsa, Vt_lsa = np.linalg.svd(doc_term, full_matrices=False)

# Project docs into 2D latent space (top-2 components)
k = 2
doc_embeddings = U_lsa[:, :k] @ np.diag(S_lsa[:k])

print("Document embeddings in 2D latent space:")
doc_names = ["cat+feline", "dog+canine", "cat", "python+java", "python"]
for name, emb in zip(doc_names, doc_embeddings):
    print(f"  {name:<15}: {emb.round(3)}")

# Cosine similarity in latent space
def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(f"\ncosine_sim('cat+feline', 'cat')         = {cos_sim(doc_embeddings[0], doc_embeddings[2]):.4f}")
print(f"cosine_sim('cat+feline', 'python+java') = {cos_sim(doc_embeddings[0], doc_embeddings[3]):.4f}")
print("→ SVD groups semantically similar documents even without exact word overlap.")

# ─────────────────────────────────────────
# 5. Recommender system via matrix factorization (SVD)
# ─────────────────────────────────────────
print("\n=== 5. Recommender System via SVD ===")

# User-item rating matrix (0 = not rated)
# Users: Alice, Bob, Carol, Dave
# Items: Item1..Item5
ratings = np.array([
    [5, 3, 0, 1, 0],
    [4, 0, 4, 1, 2],
    [1, 1, 0, 5, 4],
    [0, 2, 3, 3, 0],
], dtype=float)

# Fill missing ratings with column (item) mean before SVD
col_means = np.where(ratings != 0, ratings, np.nan)
col_means = np.nanmean(col_means, axis=0)
filled = ratings.copy()
for j in range(filled.shape[1]):
    filled[filled[:, j] == 0, j] = col_means[j]

U_r, S_r, Vt_r = np.linalg.svd(filled, full_matrices=False)

# Low-rank approximation with k=2 latent factors
k = 2
predicted = U_r[:, :k] @ np.diag(S_r[:k]) @ Vt_r[:k, :]
predicted = np.clip(predicted, 1, 5)   # ratings are 1-5

print("Original ratings (0 = not rated):")
print(ratings)
print("\nSVD-predicted ratings:")
print(predicted.round(2))

# Alice's missing ratings were at index [0,2] and [0,4]
print(f"\nAlice's predicted rating for Item3 (unrated): {predicted[0,2]:.2f}")
print(f"Alice's predicted rating for Item5 (unrated): {predicted[0,4]:.2f}")
print("→ SVD discovers latent factors (e.g. genre preferences) to fill gaps.")
