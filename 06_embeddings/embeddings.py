"""
06_embeddings/embeddings.py
============================
Embeddings — when meaning becomes a vector.

Covers:
- What an embedding is
- Cosine similarity between embeddings
- King − Man + Woman ≈ Queen (analogy arithmetic)
- Building a tiny word embedding from co-occurrence matrix + SVD
- Nearest-neighbour search in embedding space
"""

import numpy as np

# ─────────────────────────────────────────
# 1. Pre-trained-style embeddings (manually crafted for illustration)
# ─────────────────────────────────────────
print("=== 1. Embedding Basics ===")

# 6-dimensional embeddings (in reality BERT uses 768-dim, GPT uses 1600-dim etc.)
# Dimensions loosely represent: [royalty, gender_m, gender_f, animate, tech, nature]
embeddings = {
    "king":    np.array([ 0.9,  0.8, -0.1,  0.7, 0.0,  0.0]),
    "queen":   np.array([ 0.9, -0.1,  0.8,  0.7, 0.0,  0.0]),
    "man":     np.array([ 0.1,  0.9, -0.2,  0.8, 0.1,  0.0]),
    "woman":   np.array([ 0.1, -0.2,  0.9,  0.8, 0.1,  0.0]),
    "prince":  np.array([ 0.7,  0.7, -0.1,  0.6, 0.0,  0.0]),
    "apple":   np.array([-0.1,  0.0,  0.0,  0.0, 0.3,  0.8]),
    "python":  np.array([-0.1,  0.0,  0.0,  0.5, 0.9, -0.1]),
    "java":    np.array([-0.1,  0.0,  0.0,  0.0, 0.95, 0.0]),
}

for word, vec in embeddings.items():
    print(f"  {word:<8}: {vec}")

# ─────────────────────────────────────────
# 2. Cosine similarity
# ─────────────────────────────────────────
print("\n=== 2. Cosine Similarity ===")

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

pairs = [("king", "queen"), ("king", "prince"), ("king", "python"), ("python", "java")]
for w1, w2 in pairs:
    sim = cosine_similarity(embeddings[w1], embeddings[w2])
    print(f"  sim({w1:<8}, {w2:<8}) = {sim:.4f}")

# ─────────────────────────────────────────
# 3. Analogy arithmetic: King − Man + Woman ≈ Queen
# ─────────────────────────────────────────
print("\n=== 3. Analogy Arithmetic: king − man + woman ≈ ? ===")

result_vec = embeddings["king"] - embeddings["man"] + embeddings["woman"]

# Find nearest neighbour (excluding the query words)
exclude = {"king", "man", "woman"}
scores = {
    word: cosine_similarity(result_vec, vec)
    for word, vec in embeddings.items()
    if word not in exclude
}
ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

print(f"  king − man + woman = {result_vec.round(3)}")
print(f"\n  Nearest neighbours:")
for word, score in ranked:
    print(f"    {word:<10}: {score:.4f}")
print(f"\n  → Top result: '{ranked[0][0]}'  (expected: queen)")

# ─────────────────────────────────────────
# 4. Build word embeddings from scratch via co-occurrence + SVD
# ─────────────────────────────────────────
print("\n=== 4. Word Embeddings from Co-occurrence Matrix + SVD ===")

corpus = [
    "the cat sat on the mat",
    "the cat ate the rat",
    "the dog sat on the log",
    "the dog chased the cat",
    "the rat ran from the cat",
    "the dog ate the bone",
]

# Build vocabulary
words_all = " ".join(corpus).split()
vocab = sorted(set(words_all))
word2idx = {w: i for i, w in enumerate(vocab)}
V = len(vocab)

print(f"  Vocabulary ({V} words): {vocab}")

# Build co-occurrence matrix (window size = 2)
window = 2
cooc = np.zeros((V, V))

for sentence in corpus:
    tokens = sentence.split()
    for i, target in enumerate(tokens):
        for j in range(max(0, i - window), min(len(tokens), i + window + 1)):
            if i != j:
                cooc[word2idx[target], word2idx[tokens[j]]] += 1

print(f"\n  Co-occurrence matrix shape: {cooc.shape}")

# Apply SVD — keep top 3 dimensions
U_cooc, S_cooc, Vt_cooc = np.linalg.svd(cooc, full_matrices=False)
dim = 3
word_vecs = U_cooc[:, :dim] @ np.diag(S_cooc[:dim])   # (V, dim)

print(f"\n  Word vectors (dim={dim}):")
for word in vocab:
    vec = word_vecs[word2idx[word]]
    print(f"    {word:<8}: {vec.round(3)}")

# Check similarity in learnt space
print("\n  Similarities in learnt embedding space:")
for w1, w2 in [("cat", "rat"), ("cat", "dog"), ("cat", "bone")]:
    sim = cosine_similarity(word_vecs[word2idx[w1]], word_vecs[word2idx[w2]])
    print(f"    sim({w1}, {w2}) = {sim:.4f}")

# ─────────────────────────────────────────
# 5. Nearest-neighbour search
# ─────────────────────────────────────────
print("\n=== 5. Nearest-Neighbour Search ===")

def nearest_neighbours(query_word, word_vectors, word_index, top_k=3):
    query_vec = word_vectors[word_index[query_word]]
    scores = {}
    for word, idx in word_index.items():
        if word == query_word:
            continue
        scores[word] = cosine_similarity(query_vec, word_vectors[idx])
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

for query in ["cat", "dog"]:
    neighbours = nearest_neighbours(query, word_vecs, word2idx)
    print(f"  Nearest to '{query}': {[(w, round(s,4)) for w, s in neighbours]}")
