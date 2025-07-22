"""
Multi‑task CNN‑Transformer with attention heatmap, Grad‑CAM and full weight saving
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderLayerWAttn(nn.TransformerEncoderLayer):
    """Transformer encoder layer that can optionally return attention weights."""
    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False, return_attn=False):
        x = src
        attn_output, attn_w = self.self_attn(
            x, x, x,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=return_attn,
            is_causal=is_causal,
            average_attn_weights=False
        )
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        z = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(z)
        out = self.norm2(x)

        if return_attn:
            return out, attn_w.mean(0).detach()
        else:
            return out


class Backbone(nn.Module):
    """CNN + Transformer backbone"""
    def __init__(self, d_model=128, n_heads=4, n_layers=3, seq_len=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),  # 512 → 128
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4)   # 128 → 32
        )
        self.layers = nn.ModuleList([
            EncoderLayerWAttn(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=512,
                dropout=0.1,
                batch_first=False
            ) for _ in range(n_layers)
        ])
        self.seq_len = seq_len

    def forward(self, x, return_attn=False):
        x = self.conv(x.unsqueeze(1))     # [B, 128, 32]
        x = x.permute(2, 0, 1)            # [32, B, 128]
        attn_out = []
        for layer in self.layers:
            if return_attn:
                x, w = layer(x, return_attn=True)
                attn_out.append(w)       # [n_heads, seq, seq]
            else:
                x = layer(x)
        feat = x.permute(1, 0, 2).contiguous().view(x.size(1), -1)  # [B, seq*d_model]
        return (feat, attn_out) if return_attn else feat


class Head(nn.Module):
    """Classification head"""
    def __init__(self, in_dim, out_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, out_classes)
        )

    def forward(self, x):
        return self.net(x)


class MultiTaskCNNTransformer(nn.Module):
    """Full multi-task model"""
    def __init__(self, seq_len=32, d_model=128):
        super().__init__()
        self.backbone = Backbone(d_model=d_model, seq_len=seq_len)
        self.head_patient = Head(seq_len * d_model, 2)
        self.head_stage = Head(seq_len * d_model, 5)
        self.head_metastasis = Head(seq_len * d_model, 2)

    def forward(self, x, return_attn=False):
        feat, attn = self.backbone(x, return_attn=return_attn) if return_attn else (self.backbone(x), [])
        out_p = self.head_patient(feat)
        out_s = self.head_stage(feat)
        out_m = self.head_metastasis(feat)
        return (out_p, out_s, out_m, attn) if return_attn else (out_p, out_s, out_m)


def masked_multitask_loss(outputs, targets, criterions, weights=(1., 1., 1.)):
    """Compute weighted loss for multi-task with optional -1 masking."""
    out_p, out_s, out_m = outputs
    y_p, y_s, y_m = targets
    crit_p, crit_s, crit_m = criterions

    loss_p = crit_p(out_p, y_p).mean()

    mask_s = y_s != -1
    loss_s = crit_s(out_s[mask_s], y_s[mask_s]).mean() if mask_s.any() else 0.

    mask_m = y_m != -1
    loss_m = crit_m(out_m[mask_m], y_m[mask_m]).mean() if mask_m.any() else 0.

    return weights[0]*loss_p + weights[1]*loss_s + weights[2]*loss_m


def gradcam_1d(model, signal, task='patient', class_idx=1, conv_name='backbone.conv.0'):
    """
    Compute 1D Grad-CAM over the first conv layer
    """
    model.eval()
    activations = {}
    gradients = {}

    def fwd_hook(_, __, output):
        activations['value'] = output.detach()

    def bwd_hook(_, __, grad_output):
        gradients['value'] = grad_output[0].detach()

    layer = dict(model.named_modules())[conv_name]
    h1 = layer.register_forward_hook(fwd_hook)
    h2 = layer.register_full_backward_hook(bwd_hook)

    signal = signal.clone().detach().requires_grad_(True)
    out_p, out_s, out_m = model(signal)
    if task == 'patient':
        score = out_p[:, class_idx].sum()
    elif task == 'stage':
        score = out_s[:, class_idx].sum()
    else:
        score = out_m[:, class_idx].sum()

    model.zero_grad()
    score.backward()

    w = gradients['value'].mean(dim=2, keepdim=True)    # [B,C,1]
    cam = F.relu((w * activations['value']).sum(dim=1)).squeeze()
    cam -= cam.min()
    cam /= cam.max() + 1e-8
    h1.remove()
    h2.remove()
    return cam.cpu().numpy()


if __name__ == "__main__":
    # Example usage & export weights
    model = MultiTaskCNNTransformer()
    dummy_input = torch.randn(2, 512)
    out_p, out_s, out_m, attn = model(dummy_input, return_attn=True)

    print(f"out_p shape: {out_p.shape}")
    print(f"out_s shape: {out_s.shape}")
    print(f"out_m shape: {out_m.shape}")
    print(f"attn layers: {len(attn)}")

    torch.save(model.state_dict(), "cnn_transformer_weights.pth")
    torch.save(model, "cnn_transformer_full.pth")
    print("✅ Model weights saved: cnn_transformer_weights.pth & cnn_transformer_full.pth")
