"""
03_dot_products/dot_products.py
================================
Dot Products — the engine of neural networks.

Covers:
- Computing dot products
- Geometric interpretation (cosine angle)
- Similarity: positive / zero / negative
- Cosine similarity search
- Simulated neuron using dot product
- Attention score preview
"""

import numpy as np

# ─────────────────────────────────────────
# 1. Basic dot product
# ─────────────────────────────────────────
print("=== 1. Dot Product ===")
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])

dot = np.dot(a, b)
manual = sum(ai * bi for ai, bi in zip(a, b))

print(f"a = {a}")
print(f"b = {b}")
print(f"np.dot(a, b)     = {dot}")
print(f"Manual Σ aᵢbᵢ   = {manual}")

# ─────────────────────────────────────────
# 2. Geometric interpretation: a·b = |a||b|cos(θ)
# ─────────────────────────────────────────
print("\n=== 2. Geometric Interpretation ===")

def angle_between(a, b):
    cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)   # numerical safety
    return np.degrees(np.arccos(cos_theta))

same_dir  = np.array([1.0, 0.0])
perp      = np.array([0.0, 1.0])
opposite  = np.array([-1.0, 0.0])

print(f"Angle(same direction) = {angle_between(same_dir, same_dir):.1f}°  → dot = {np.dot(same_dir, same_dir):.1f}")
print(f"Angle(perpendicular)  = {angle_between(same_dir, perp):.1f}°  → dot = {np.dot(same_dir, perp):.1f}")
print(f"Angle(opposite)       = {angle_between(same_dir, opposite):.1f}° → dot = {np.dot(same_dir, opposite):.1f}")

# ─────────────────────────────────────────
# 3. Cosine similarity for ML vectors
# ─────────────────────────────────────────
print("\n=== 3. Cosine Similarity — Embedding Search ===")

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Fake 4-dim embeddings for 5 words
embeddings = {
    "king":   np.array([ 0.9,  0.1,  0.8,  0.3]),
    "queen":  np.array([ 0.85, 0.15, 0.75, 0.6]),
    "man":    np.array([ 0.7,  0.1,  0.2,  0.1]),
    "woman":  np.array([ 0.65, 0.15, 0.18, 0.4]),
    "apple":  np.array([-0.3,  0.9, -0.1,  0.2]),
}

query = embeddings["king"]
print(f"Query: 'king'")
for word, vec in embeddings.items():
    sim = cosine_similarity(query, vec)
    print(f"  similarity(king, {word:<6}) = {sim:.4f}")

# ─────────────────────────────────────────
# 4. Simulated neuron: output = activation(W · x + b)
# ─────────────────────────────────────────
print("\n=== 4. Single Neuron Simulation ===")

np.random.seed(0)
input_size = 5

W = np.random.randn(input_size)   # weight vector
b = 0.1                            # bias scalar
x = np.array([0.5, -0.3, 0.8, 0.1, -0.6])  # input

raw = np.dot(W, x) + b            # the dot product

relu    = lambda z: max(0, z)
sigmoid = lambda z: 1 / (1 + np.exp(-z))

print(f"x       = {x}")
print(f"W       = {W.round(3)}")
print(f"W · x + b = {raw:.4f}")
print(f"ReLU output   = {relu(raw):.4f}")
print(f"Sigmoid output = {sigmoid(raw):.4f}")
print("→ Every neuron in a network computes exactly one dot product.")

# ─────────────────────────────────────────
# 5. Dot product as attention score (preview)
# ─────────────────────────────────────────
print("\n=== 5. Attention Score Preview (Q · Kᵀ) ===")

d_k = 4  # key/query dimension
np.random.seed(1)

Q = np.random.randn(3, d_k)   # 3 query vectors
K = np.random.randn(5, d_k)   # 5 key vectors

# Raw attention scores: each query dotted with every key
scores = Q @ K.T               # (3, 5)
scaled = scores / np.sqrt(d_k) # scale to avoid exploding softmax

def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

weights = softmax(scaled)      # (3, 5) — each row sums to 1

print(f"Q shape: {Q.shape}  (3 queries, dim={d_k})")
print(f"K shape: {K.shape}  (5 keys,   dim={d_k})")
print(f"Attention weight matrix shape: {weights.shape}")
print(f"\nAttention weights for query 0:\n  {weights[0].round(4)}")
print(f"Sum = {weights[0].sum():.4f}  ← always 1")
print("→ Attention is just dot products + softmax. Pure linear algebra.")
