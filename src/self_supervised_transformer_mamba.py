import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Utility: Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0) 
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Transformer Encoder with Self-Supervised Pretraining
class PhotoacousticTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, n_heads=4, n_layers=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_enc = PositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Reconstruction Head
        self.reconstruct = nn.Linear(hidden_dim, input_dim)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 1)
        )

    def forward(self, x, mask=None, task='reconstruct'):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)

        if task == 'reconstruct':
            out = self.reconstruct(x)
        elif task == 'classify':
            pooled = x.mean(dim=1)
            out = self.classifier(pooled).squeeze(-1)
        else:
            raise ValueError("task must be 'reconstruct' or 'classify'")
        return out

def masked_reconstruction_loss(y_pred, y_true, mask):
    """
    Only compute reconstruction loss on masked positions
    """
    loss = F.mse_loss(y_pred[mask], y_true[mask])
    return loss

# PAFCMamba: State Space Model with Self-supervised and Classification
class MambaBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.U = nn.Linear(hidden_dim, hidden_dim)
        self.sigmoid_gate = nn.Sigmoid()
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        h = self.W(x) * self.sigmoid_gate(self.U(x))
        return self.norm(x + h)

class PAFCMamba(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, n_heads=4, n_layers=3, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_enc = PositionalEncoding(hidden_dim)
        self.mamba_blocks = nn.ModuleList([MambaBlock(hidden_dim) for _ in range(n_layers)])
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)

        # Reconstruction Head
        self.reconstruct = nn.Linear(hidden_dim, input_dim)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 1)
        )

    def forward(self, x, mask=None, task='reconstruct'):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        for block in self.mamba_blocks:
            x = block(x)
        x, _ = self.attn(x, x, x)

        if task == 'reconstruct':
            out = self.reconstruct(x)
        elif task == 'classify':
            pooled = x.mean(dim=1)
            out = self.classifier(pooled).squeeze(-1)
        else:
            raise ValueError("task must be 'reconstruct' or 'classify'")
        return out
