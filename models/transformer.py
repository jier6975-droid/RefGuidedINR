"""
Transformer Model Implementation
Based on "Attention Is All You Need" (Vaswani et al., 2017)
https://arxiv.org/abs/1706.03762

This is a complete implementation of the Transformer architecture including:
- Multi-Head Attention
- Position-wise Feed-Forward Networks
- Positional Encoding
- Encoder and Decoder stacks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    Inject positional information into token embeddings using sine and cosine functions.
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.
    Allows the model to jointly attend to information from different representation subspaces.
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Compute scaled dot-product attention.
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V

        Args:
            Q, K, V: Query, Key, Value tensors of shape (batch_size, num_heads, seq_len, d_k)
            mask: Optional mask tensor
        Returns:
            attention_output: Tensor of shape (batch_size, num_heads, seq_len, d_k)
            attention_weights: Tensor of shape (batch_size, num_heads, seq_len, seq_len)
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        attention_output = torch.matmul(attention_weights, V)
        return attention_output, attention_weights

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query, key, value: Tensors of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor
        Returns:
            output: Tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size = query.size(0)

        # Linear projections and reshape to (batch_size, num_heads, seq_len, d_k)
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Apply attention
        attention_output, self.attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads and apply output projection
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(attention_output)

        return output


class PositionWiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """
    Single Encoder layer consisting of:
    1. Multi-Head Self-Attention
    2. Position-wise Feed-Forward Network
    Both with residual connections and layer normalization.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor
        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        # Self-attention with residual connection and layer norm
        attention_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention_output))

        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class DecoderLayer(nn.Module):
    """
    Single Decoder layer consisting of:
    1. Masked Multi-Head Self-Attention
    2. Multi-Head Cross-Attention (attending to encoder output)
    3. Position-wise Feed-Forward Network
    All with residual connections and layer normalization.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: Tensor of shape (batch_size, tgt_seq_len, d_model)
            encoder_output: Tensor of shape (batch_size, src_seq_len, d_model)
            src_mask: Optional source mask
            tgt_mask: Optional target mask (for masking future positions)
        Returns:
            Tensor of shape (batch_size, tgt_seq_len, d_model)
        """
        # Masked self-attention with residual connection and layer norm
        self_attention_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attention_output))

        # Cross-attention with encoder output
        cross_attention_output = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attention_output))

        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x


class Encoder(nn.Module):
    """
    Transformer Encoder consisting of a stack of N encoder layers.
    """
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len) containing token indices
            mask: Optional mask tensor
        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)

        for layer in self.layers:
            x = layer(x, mask)

        return x


class Decoder(nn.Module):
    """
    Transformer Decoder consisting of a stack of N decoder layers.
    """
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: Tensor of shape (batch_size, tgt_seq_len) containing token indices
            encoder_output: Tensor of shape (batch_size, src_seq_len, d_model)
            src_mask: Optional source mask
            tgt_mask: Optional target mask
        Returns:
            Tensor of shape (batch_size, tgt_seq_len, d_model)
        """
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)

        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return x


class Transformer(nn.Module):
    """
    Complete Transformer model for sequence-to-sequence tasks.
    Based on "Attention Is All You Need" (Vaswani et al., 2017)
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_layers=6,
                 num_heads=8, d_ff=2048, max_len=5000, dropout=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(src_vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout)
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        # Initialize parameters with Glorot / fan_avg
        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz):
        """
        Generate a mask to prevent attending to future positions.
        Used in decoder self-attention.
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return ~mask

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Args:
            src: Source sequence tensor of shape (batch_size, src_seq_len)
            tgt: Target sequence tensor of shape (batch_size, tgt_seq_len)
            src_mask: Optional source mask
            tgt_mask: Optional target mask
        Returns:
            Tensor of shape (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        output = self.output_projection(decoder_output)
        return output


def count_parameters(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Example usage and model size test
    src_vocab_size = 10000
    tgt_vocab_size = 10000

    # Create model with default parameters from the paper
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=512,
        num_layers=6,
        num_heads=8,
        d_ff=2048,
        max_len=5000,
        dropout=0.1
    )

    print("=" * 60)
    print("Transformer Model Architecture")
    print("=" * 60)
    print(f"Source Vocabulary Size: {src_vocab_size}")
    print(f"Target Vocabulary Size: {tgt_vocab_size}")
    print(f"Model Dimension (d_model): 512")
    print(f"Number of Layers: 6")
    print(f"Number of Attention Heads: 8")
    print(f"Feed-Forward Dimension (d_ff): 2048")
    print(f"Total Trainable Parameters: {count_parameters(model):,}")
    print("=" * 60)

    # Test forward pass
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 12

    src = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len))

    # Generate target mask to prevent attending to future positions
    tgt_mask = model.generate_square_subsequent_mask(tgt_seq_len).unsqueeze(0).unsqueeze(0)

    output = model(src, tgt, tgt_mask=tgt_mask)
    print(f"\nTest forward pass:")
    print(f"Input source shape: {src.shape}")
    print(f"Input target shape: {tgt.shape}")
    print(f"Output shape: {output.shape}")
    print("=" * 60)
