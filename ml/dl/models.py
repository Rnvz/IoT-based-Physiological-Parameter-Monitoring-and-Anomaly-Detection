"""
ml.dl.models: Compact Autoencoder and Variational Autoencoder architectures for physiological anomaly detection.

Specifications:
- Total parameter count under 10,000 (suitable for edge microcontrollers).
- Supports input dimensions: 3 (v3: HR, SpO2, Temp) and 2 (v2: HR, SpO2).
- Anomaly score: Reconstruction Mean Squared Error (MSE).
- NumPy forward fallback implemented for deployment portability without PyTorch/ONNX runtime.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np

try:
    import torch
    import torch.nn as nn
    _Module = nn.Module
except ImportError:
    torch = None
    nn = None
    _Module = object


class DenseAutoencoder(_Module):
    """
    Compact symmetric Autoencoder:
    Encoder: input_dim -> 16 -> 8 -> 2 (bottleneck)
    Decoder: 2 -> 8 -> 16 -> input_dim
    Total parameters: ~540 (well under 10,000 limit).
    """

    def __init__(self, input_dim: int = 3, bottleneck_dim: int = 2, dropout: float = 0.05):
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(16, 8),
            nn.LayerNorm(8),
            nn.GELU(),
            nn.Linear(8, bottleneck_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 8),
            nn.LayerNorm(8),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(8, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Computes per-sample MSE reconstruction error."""
        self.eval()
        with torch.no_grad():
            x_rec = self.forward(x)
            mse = torch.mean((x - x_rec) ** 2, dim=-1)
        return mse


class DenseVAE(_Module):
    """
    Compact Variational Autoencoder:
    Encoder: input_dim -> 16 -> 8 -> mu(2), logvar(2)
    Decoder: 2 -> 8 -> 16 -> input_dim
    Anomaly score: Reconstruction MSE error (per pre-registration protocol).
    """

    def __init__(self, input_dim: int = 3, bottleneck_dim: int = 2, dropout: float = 0.05):
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim

        self.encoder_backbone = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(16, 8),
            nn.LayerNorm(8),
            nn.GELU(),
        )

        self.fc_mu = nn.Linear(8, bottleneck_dim)
        self.fc_logvar = nn.Linear(8, bottleneck_dim)

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 8),
            nn.LayerNorm(8),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(8, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Linear(16, input_dim)
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_backbone(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_rec = self.decode(z)
        return x_rec, mu, logvar

    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Computes per-sample MSE reconstruction error."""
        self.eval()
        with torch.no_grad():
            x_rec, _, _ = self.forward(x)
            mse = torch.mean((x - x_rec) ** 2, dim=-1)
        return mse


def count_parameters(model: nn.Module) -> int:
    """Returns total trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ==============================================================================
# NUMPY FORWARD FALLBACK IMPLEMENTATION
# ==============================================================================

from scipy.special import erf


def gelu_np(x: np.ndarray) -> np.ndarray:
    """Exact Gaussian Error Linear Unit in NumPy matching torch.nn.GELU()."""
    return (0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))).astype(np.float32)


def layer_norm_np(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Layer normalization in NumPy across last dimension."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return x_norm * weight + bias


class NumPyAutoencoderForward:
    """
    Pure NumPy forward pass for DenseAutoencoder.
    Guarantees deployment without PyTorch or ONNX Runtime.
    """

    def __init__(self, weights: Dict[str, np.ndarray]):
        self.weights = weights

    @classmethod
    def from_torch_model(cls, torch_model: DenseAutoencoder) -> "NumPyAutoencoderForward":
        weights = {}
        for k, v in torch_model.state_dict().items():
            weights[k] = v.cpu().numpy()
        return cls(weights)

    def forward(self, x: np.ndarray) -> np.ndarray:
        w = self.weights
        # Encoder
        h = x @ w["encoder.0.weight"].T + w["encoder.0.bias"]
        h = layer_norm_np(h, w["encoder.1.weight"], w["encoder.1.bias"])
        h = gelu_np(h)
        h = h @ w["encoder.4.weight"].T + w["encoder.4.bias"]
        h = layer_norm_np(h, w["encoder.5.weight"], w["encoder.5.bias"])
        h = gelu_np(h)
        z = h @ w["encoder.7.weight"].T + w["encoder.7.bias"]

        # Decoder
        d = z @ w["decoder.0.weight"].T + w["decoder.0.bias"]
        d = layer_norm_np(d, w["decoder.1.weight"], w["decoder.1.bias"])
        d = gelu_np(d)
        d = d @ w["decoder.4.weight"].T + w["decoder.4.bias"]
        d = layer_norm_np(d, w["decoder.5.weight"], w["decoder.5.bias"])
        d = gelu_np(d)
        x_rec = d @ w["decoder.7.weight"].T + w["decoder.7.bias"]
        return x_rec

    def score(self, x: np.ndarray) -> np.ndarray:
        x_rec = self.forward(x)
        return np.mean((x - x_rec) ** 2, axis=-1)
