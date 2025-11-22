"""
GPT Model Architecture
Based on "Attention Is All You Need" paper and GPT-2
Adapted from Sebastian Raschka's LLMs-from-scratch
"""

import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism
    This is the "magic" that lets the model understand relationships between words
    """
    def __init__(self, d_in, d_out, context_length, num_heads, dropout=0.0):
        super().__init__()
        
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Dimension per head
        
        # Linear projections for queries, keys, values
        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.W_key = nn.Linear(d_in, d_out, bias=False)
        self.W_value = nn.Linear(d_in, d_out, bias=False)
        self.out_proj = nn.Linear(d_out, d_out)
        
        self.dropout = nn.Dropout(dropout)
        
        # Causal mask (prevents looking at future words)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
    
    def forward(self, x):
        batch_size, num_tokens, d_in = x.shape
        
        # Apply linear transformations
        queries = self.W_query(x)  # (batch, tokens, d_out)
        keys = self.W_key(x)
        values = self.W_value(x)
        
        # Split into multiple heads
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        values = values.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        queries = queries.transpose(1, 2)  # (batch, heads, tokens, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # Compute attention scores
        attention_scores = queries @ keys.transpose(2, 3)  # (batch, heads, tokens, tokens)
        
        # Scale
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        
        # Apply causal mask
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attention_scores = attention_scores.masked_fill(mask_bool, -torch.inf)
        
        # Softmax
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context_vector = attention_weights @ values  # (batch, heads, tokens, head_dim)
        
        # Combine heads
        context_vector = context_vector.transpose(1, 2).contiguous()
        context_vector = context_vector.view(batch_size, num_tokens, self.d_out)
        
        # Final projection
        context_vector = self.out_proj(context_vector)
        
        return context_vector


class FeedForward(nn.Module):
    """
    Feed-forward neural network
    Processes each token independently after attention
    """
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.GELU(),  # Activation function
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )
    
    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    """
    One transformer block = Attention + Feed-forward
    GPT model stacks multiple of these blocks
    """
    def __init__(self, cfg):
        super().__init__()
        
        # Multi-head attention
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"]
        )
        
        # Feed-forward network
        self.ff = FeedForward(cfg)
        
        # Layer normalization (stabilizes training)
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        
        # Dropout for regularization
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
    
    def forward(self, x):
        # Attention block with residual connection
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Residual connection
        
        # Feed-forward block with residual connection
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        
        return x


class GPTModel(nn.Module):
    """
    Complete GPT Model
    This is what you'll train on Telugu text
    """
    def __init__(self, cfg):
        super().__init__()
        
        # Token embeddings (convert token IDs to vectors)
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        
        # Position embeddings (where is this token in the sequence?)
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        
        # Dropout
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        # Stack of transformer blocks
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(cfg["emb_dim"])
        
        # Output layer (predict next token)
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
    
    def forward(self, in_idx):
        """
        Forward pass
        
        Parameters:
        -----------
        in_idx : torch.Tensor
            Input token IDs, shape (batch_size, sequence_length)
        
        Returns:
        --------
        logits : torch.Tensor
            Predictions for next token, shape (batch_size, sequence_length, vocab_size)
        """
        batch_size, seq_len = in_idx.shape
        
        # Token embeddings
        tok_embeds = self.tok_emb(in_idx)
        
        # Position embeddings
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        
        # Combine
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        
        # Pass through transformer blocks
        x = self.trf_blocks(x)
        
        # Final norm
        x = self.final_norm(x)
        
        # Output logits
        logits = self.out_head(x)
        
        return logits


# Configuration for Telugu GPT
TELUGU_GPT_CONFIG = {
    "vocab_size": 32000,        # Size of Telugu tokenizer vocabulary
    "context_length": 512,      # Maximum sequence length
    "emb_dim": 768,            # Embedding dimension
    "n_heads": 12,             # Number of attention heads
    "n_layers": 12,            # Number of transformer blocks
    "drop_rate": 0.1,          # Dropout rate
}


# Example: Create the model
if __name__ == "__main__":
    model = GPTModel(TELUGU_GPT_CONFIG)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Telugu GPT Model created!")
    print(f"📊 Total parameters: {total_params:,}")
    print(f"📊 Configuration: {TELUGU_GPT_CONFIG}")
    
    # Test forward pass
    test_input = torch.randint(0, 32000, (2, 10))  # Batch of 2, sequence length 10
    output = model(test_input)
    print(f"✅ Test forward pass successful!")
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
