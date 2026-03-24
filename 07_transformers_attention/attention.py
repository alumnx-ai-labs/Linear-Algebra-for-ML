"""
07_transformers_attention/attention.py
=======================================
Transformers & Attention — linear algebra in action.

Covers:
- Scaled dot-product attention from scratch (no PyTorch/TensorFlow)
- Multi-head attention from scratch
- Positional encoding
- A full tiny Transformer encoder block
- Walkthrough of every matrix operation with shape annotations
"""

import numpy as np

# ─────────────────────────────────────────
# Utility
# ─────────────────────────────────────────
def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def layer_norm(x, eps=1e-6):
    mean = x.mean(axis=-1, keepdims=True)
    std  = x.std(axis=-1,  keepdims=True)
    return (x - mean) / (std + eps)

# ─────────────────────────────────────────
# 1. Scaled Dot-Product Attention
#    Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V
# ─────────────────────────────────────────
def scaled_dot_product_attention(Q, K, V, mask=None, verbose=False):
    """
    Q : (seq_len, d_k)
    K : (seq_len, d_k)
    V : (seq_len, d_v)
    Returns: output (seq_len, d_v), attention_weights (seq_len, seq_len)
    """
    d_k = Q.shape[-1]

    # Step 1: Q · Kᵀ → raw scores
    scores = Q @ K.T                        # (seq_len, seq_len)

    # Step 2: Scale by √d_k to prevent exploding gradients
    scores = scores / np.sqrt(d_k)          # (seq_len, seq_len)

    # Step 3: Optional causal mask (for decoder)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)

    # Step 4: Softmax → attention weights (each row sums to 1)
    weights = softmax(scores, axis=-1)      # (seq_len, seq_len)

    # Step 5: Weighted sum of values
    output = weights @ V                    # (seq_len, d_v)

    if verbose:
        print(f"  Q shape    : {Q.shape}")
        print(f"  K shape    : {K.shape}")
        print(f"  V shape    : {V.shape}")
        print(f"  scores Q·Kᵀ: {scores.shape}  (before softmax)")
        print(f"  weights    : {weights.shape}  (after softmax, rows sum to 1)")
        print(f"  output     : {output.shape}")

    return output, weights


print("=== 1. Scaled Dot-Product Attention ===")
np.random.seed(42)

seq_len = 4   # 4 tokens
d_model = 8   # embedding dimension
d_k     = 4   # query/key dimension

# Random token embeddings for 4 tokens
X = np.random.randn(seq_len, d_model)    # (4, 8)

# Linear projections W_Q, W_K, W_V (normally learnt weights)
W_Q = np.random.randn(d_model, d_k) * 0.1
W_K = np.random.randn(d_model, d_k) * 0.1
W_V = np.random.randn(d_model, d_k) * 0.1

Q = X @ W_Q   # (4, 4)
K = X @ W_K   # (4, 4)
V = X @ W_V   # (4, 4)

output, weights = scaled_dot_product_attention(Q, K, V, verbose=True)

print(f"\n  Attention weights (how much each token attends to others):")
print(weights.round(3))
print(f"  Row sums: {weights.sum(axis=1).round(4)}  ← each sums to 1.0")

# ─────────────────────────────────────────
# 2. Multi-Head Attention
# ─────────────────────────────────────────
print("\n=== 2. Multi-Head Attention ===")

class MultiHeadAttention:
    def __init__(self, d_model, num_heads, seed=0):
        assert d_model % num_heads == 0
        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads

        rng = np.random.default_rng(seed)
        scale = 0.1
        # One projection matrix per head
        self.W_Qs = [rng.standard_normal((d_model, self.d_k)) * scale for _ in range(num_heads)]
        self.W_Ks = [rng.standard_normal((d_model, self.d_k)) * scale for _ in range(num_heads)]
        self.W_Vs = [rng.standard_normal((d_model, self.d_k)) * scale for _ in range(num_heads)]
        # Output projection
        self.W_O  = rng.standard_normal((d_model, d_model)) * scale

    def forward(self, X):
        """X: (seq_len, d_model)"""
        head_outputs = []
        for h in range(self.num_heads):
            Q_h = X @ self.W_Qs[h]   # (seq_len, d_k)
            K_h = X @ self.W_Ks[h]
            V_h = X @ self.W_Vs[h]
            out_h, _ = scaled_dot_product_attention(Q_h, K_h, V_h)
            head_outputs.append(out_h)

        # Concatenate all heads along last dimension
        concat = np.concatenate(head_outputs, axis=-1)   # (seq_len, d_model)

        # Final linear projection
        return concat @ self.W_O                          # (seq_len, d_model)


mha = MultiHeadAttention(d_model=8, num_heads=2)
mha_output = mha.forward(X)
print(f"  Input  shape: {X.shape}          (seq_len=4, d_model=8)")
print(f"  Output shape: {mha_output.shape}  (same — attention is dimension-preserving)")

# ─────────────────────────────────────────
# 3. Causal (masked) attention — for autoregressive decoding
# ─────────────────────────────────────────
print("\n=== 3. Causal Mask (Decoder Self-Attention) ===")

# Lower-triangular mask: token i can only attend to tokens 0..i
causal_mask = np.tril(np.ones((seq_len, seq_len)))
print(f"  Causal mask:\n{causal_mask}")

causal_out, causal_w = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
print(f"\n  Causal attention weights:")
print(causal_w.round(3))
print("  (upper triangle is ~0 — future tokens are masked)")

# ─────────────────────────────────────────
# 4. Positional Encoding
# ─────────────────────────────────────────
print("\n=== 4. Positional Encoding ===")

def positional_encoding(seq_len, d_model):
    """Sinusoidal positional encoding (Vaswani et al. 2017)."""
    PE = np.zeros((seq_len, d_model))
    pos = np.arange(seq_len)[:, np.newaxis]           # (seq_len, 1)
    div = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    PE[:, 0::2] = np.sin(pos * div)
    PE[:, 1::2] = np.cos(pos * div)
    return PE

PE = positional_encoding(seq_len=6, d_model=8)
print(f"  Positional encoding shape: {PE.shape}")
print(f"  PE[0] (token 0): {PE[0].round(3)}")
print(f"  PE[1] (token 1): {PE[1].round(3)}")
print("  → Each position gets a unique fingerprint via sine/cosine waves.")

# ─────────────────────────────────────────
# 5. Tiny Transformer Encoder Block
# ─────────────────────────────────────────
print("\n=== 5. Tiny Transformer Encoder Block ===")

def feed_forward(X, d_ff=32, seed=1):
    """Two-layer MLP applied position-wise."""
    rng = np.random.default_rng(seed)
    d_model = X.shape[-1]
    W1 = rng.standard_normal((d_model, d_ff)) * 0.1
    b1 = np.zeros(d_ff)
    W2 = rng.standard_normal((d_ff, d_model)) * 0.1
    b2 = np.zeros(d_model)
    hidden = np.maximum(0, X @ W1 + b1)   # ReLU
    return hidden @ W2 + b2

def transformer_encoder_block(X, num_heads=2):
    """
    One encoder block:
    1. Multi-Head Self-Attention
    2. Add & Layer Norm
    3. Feed-Forward Network
    4. Add & Layer Norm
    """
    d_model = X.shape[-1]

    # --- Sub-layer 1: Self-Attention ---
    mha     = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    attn_out = mha.forward(X)
    X       = layer_norm(X + attn_out)         # residual connection + norm

    # --- Sub-layer 2: Feed-Forward ---
    ff_out  = feed_forward(X)
    X       = layer_norm(X + ff_out)           # residual connection + norm

    return X


# Simulate a 4-token sequence with 8-dim embeddings + positional encoding
token_embeddings = np.random.randn(seq_len, d_model)
pe = positional_encoding(seq_len, d_model)
X_input = token_embeddings + pe               # add positional info

print(f"  Input  shape: {X_input.shape}")
encoder_output = transformer_encoder_block(X_input, num_heads=2)
print(f"  Output shape: {encoder_output.shape}")
print(f"\n  Output (first 2 tokens):\n{encoder_output[:2].round(4)}")
print("\n  Every operation inside is matrix multiplication + softmax.")
print("  Attention formula: softmax( Q·Kᵀ / √d_k ) · V")
print("  → Pure linear algebra at scale.")
