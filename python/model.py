"""
Nyströmformer Model Implementation

This module implements the Nyströmformer architecture for trading applications.
Nyströmformer uses the Nyström method to approximate self-attention with
linear O(n) complexity instead of quadratic O(n²).

Key components:
- NystromAttention: Nyström-approximated multi-head attention
- NystromEncoderLayer: Transformer encoder layer with Nyström attention
- NystromformerTrading: Complete model for price prediction and trading

References:
- Nyströmformer paper: https://arxiv.org/abs/2102.03902
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class NystromAttention(nn.Module):
    """
    Nyström-approximated multi-head self-attention.

    Instead of computing the full n×n attention matrix, this uses m landmarks
    (m << n) to approximate attention with O(n·m) complexity.

    The approximation formula:
        Ŝ = F̃ · Ã⁺ · B̃

    Where:
        F̃ = softmax(Q · K̃^T / √d)   - Full queries to landmark keys
        Ã = softmax(Q̃ · K̃^T / √d)   - Landmarks to landmarks
        B̃ = softmax(Q̃ · K^T / √d)   - Landmark queries to full keys
        Ã⁺ = Pseudoinverse of Ã

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        num_landmarks: Number of landmark points for Nyström approximation
        seq_len: Maximum sequence length
        pinv_iterations: Number of iterations for pseudoinverse approximation
        residual_conv: Whether to add residual depthwise convolution on values
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_landmarks: int = 64,
        seq_len: int = 512,
        pinv_iterations: int = 6,
        residual_conv: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert seq_len % num_landmarks == 0, "seq_len must be divisible by num_landmarks"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.num_landmarks = num_landmarks
        self.seq_len = seq_len
        self.segment_size = seq_len // num_landmarks
        self.pinv_iterations = pinv_iterations
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Optional: Residual convolution for values (skip connection)
        self.residual_conv = residual_conv
        if residual_conv:
            self.conv = nn.Conv1d(
                d_model, d_model,
                kernel_size=3,
                padding=1,
                groups=d_model  # Depthwise conv
            )

        self.dropout = nn.Dropout(dropout)

    def compute_landmarks(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute landmark points using segment-means.

        Divides the sequence into num_landmarks segments and computes
        the mean of each segment as the landmark representation.

        Args:
            x: Input tensor [batch, n_heads, seq_len, head_dim]

        Returns:
            landmarks: Tensor [batch, n_heads, num_landmarks, head_dim]
        """
        batch, n_heads, seq_len, head_dim = x.shape

        # Reshape: [batch, n_heads, num_landmarks, segment_size, head_dim]
        x_segments = x.reshape(
            batch, n_heads,
            self.num_landmarks,
            self.segment_size,
            head_dim
        )

        # Mean over segment dimension
        landmarks = x_segments.mean(dim=3)

        return landmarks

    def iterative_pinv(self, A: torch.Tensor) -> torch.Tensor:
        """
        Iterative Newton-Schulz approximation of Moore-Penrose pseudoinverse.

        More efficient than SVD on GPUs, converges in ~6 iterations.

        The iteration: Z_{k+1} = Z_k (2I - A Z_k)

        Args:
            A: Input tensor [batch, n_heads, m, m]

        Returns:
            A_pinv: Approximate pseudoinverse of A
        """
        # Initial approximation
        A_T = A.transpose(-1, -2)

        # Normalize for numerical stability
        norm_A = torch.norm(A, dim=(-2, -1), keepdim=True)
        A_normalized = A / (norm_A + 1e-6)
        A_T_normalized = A_T / (norm_A + 1e-6)

        Z = A_T_normalized

        I = torch.eye(
            A.shape[-1],
            device=A.device,
            dtype=A.dtype
        ).unsqueeze(0).unsqueeze(0)

        for _ in range(self.pinv_iterations):
            Z = 0.5 * Z @ (3 * I - A_normalized @ Z)

        # Adjust for normalization
        Z = Z / (norm_A + 1e-6)

        return Z

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Nyström attention forward pass.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            attention_mask: Optional attention mask (not fully implemented)
            return_attention: Whether to return approximate attention weights

        Returns:
            output: Output tensor [batch, seq_len, d_model]
            attention_weights: Optional approximate attention weights
        """
        batch, seq_len, _ = x.shape

        # Validate sequence length - require exact match to avoid silent errors
        # with padding and last-token selection
        if seq_len != self.seq_len:
            raise ValueError(
                f"Expected seq_len={self.seq_len}, got {seq_len}. "
                "Pad/trim inputs (and mask) before calling attention."
            )

        # Project to Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape for multi-head attention
        # [batch, seq_len, n_heads, head_dim] -> [batch, n_heads, seq_len, head_dim]
        Q = Q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Compute landmarks
        Q_landmarks = self.compute_landmarks(Q)  # [batch, n_heads, m, head_dim]
        K_landmarks = self.compute_landmarks(K)  # [batch, n_heads, m, head_dim]

        # Kernel 1: F̃ = softmax(Q @ K̃^T / √d)
        # [batch, n_heads, n, m]
        kernel_1 = F.softmax(
            torch.matmul(Q, K_landmarks.transpose(-1, -2)) * self.scale,
            dim=-1
        )

        # Kernel 2: Ã = softmax(Q̃ @ K̃^T / √d)
        # [batch, n_heads, m, m]
        kernel_2 = F.softmax(
            torch.matmul(Q_landmarks, K_landmarks.transpose(-1, -2)) * self.scale,
            dim=-1
        )

        # Kernel 3: B̃ = softmax(Q̃ @ K^T / √d)
        # [batch, n_heads, m, n]
        kernel_3 = F.softmax(
            torch.matmul(Q_landmarks, K.transpose(-1, -2)) * self.scale,
            dim=-1
        )

        # Compute pseudoinverse of kernel_2
        kernel_2_inv = self.iterative_pinv(kernel_2)

        # Efficient computation: Never materialize n×n matrix
        # Step 1: B̃ @ V → [batch, n_heads, m, head_dim]
        context_1 = torch.matmul(kernel_3, V)

        # Step 2: Ã⁺ @ (B̃ @ V) → [batch, n_heads, m, head_dim]
        context_2 = torch.matmul(kernel_2_inv, context_1)

        # Step 3: F̃ @ (Ã⁺ @ B̃ @ V) → [batch, n_heads, n, head_dim]
        output = torch.matmul(kernel_1, context_2)

        # Optional: Add residual convolution on values
        if self.residual_conv:
            V_residual = V.transpose(1, 2).reshape(batch, seq_len, -1)
            V_residual = self.conv(V_residual.transpose(1, 2)).transpose(1, 2)
            V_residual = V_residual.view(batch, seq_len, self.n_heads, self.head_dim)
            V_residual = V_residual.transpose(1, 2)
            output = output + V_residual

        # Reshape back
        output = output.transpose(1, 2).reshape(batch, seq_len, self.d_model)
        output = self.out_proj(output)
        output = self.dropout(output)

        # Optionally return attention approximation
        attention_weights = None
        if return_attention:
            # Approximate attention: F̃ @ Ã⁺ @ B̃
            attention_weights = torch.matmul(
                kernel_1,
                torch.matmul(kernel_2_inv, kernel_3)
            )

        return output, attention_weights


class NystromEncoderLayer(nn.Module):
    """
    Single Nyströmformer encoder layer.

    Consists of:
    1. Nyström self-attention with residual connection
    2. Layer normalization
    3. Feed-forward network with residual connection
    4. Layer normalization

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        num_landmarks: Number of landmark points
        seq_len: Maximum sequence length
        dim_feedforward: Feed-forward network dimension
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_landmarks: int = 64,
        seq_len: int = 512,
        dim_feedforward: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()

        self.self_attn = NystromAttention(
            d_model=d_model,
            n_heads=n_heads,
            num_landmarks=num_landmarks,
            seq_len=seq_len,
            dropout=dropout
        )

        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            return_attention: Whether to return attention weights

        Returns:
            output: Output tensor [batch, seq_len, d_model]
            attention: Optional attention weights
        """
        # Self-attention with residual
        attn_out, attention = self.self_attn(x, return_attention=return_attention)
        x = self.norm1(x + attn_out)

        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x, attention


class NystromformerTrading(nn.Module):
    """
    Complete Nyströmformer for trading applications.

    Supports multiple output types:
    - regression: Predict future returns
    - classification: Predict price direction (up/down/neutral)
    - allocation: Generate portfolio allocation weights

    Args:
        input_dim: Number of input features per timestep
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of encoder layers
        num_landmarks: Number of landmark points
        seq_len: Maximum sequence length
        output_type: Type of output ('regression', 'classification', 'allocation')
        pred_horizon: Prediction horizon for regression/allocation
        n_classes: Number of classes for classification
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_dim: int = 6,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        num_landmarks: int = 64,
        seq_len: int = 4096,
        output_type: str = 'regression',
        pred_horizon: int = 24,
        n_classes: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()

        self.output_type = output_type
        self.d_model = d_model
        self.seq_len = seq_len

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.register_buffer(
            'pos_encoding',
            self._create_pos_encoding(seq_len, d_model)
        )

        # Encoder layers
        self.layers = nn.ModuleList([
            NystromEncoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                num_landmarks=num_landmarks,
                seq_len=seq_len,
                dim_feedforward=d_model * 4,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Output head based on task type
        if output_type == 'regression':
            self.head = nn.Linear(d_model, pred_horizon)
        elif output_type == 'classification':
            self.head = nn.Linear(d_model, n_classes)
        elif output_type == 'allocation':
            self.head = nn.Sequential(
                nn.Linear(d_model, pred_horizon),
                nn.Tanh()  # Bound allocations to [-1, 1]
            )
        else:
            raise ValueError(f"Unknown output_type: {output_type}")

    def _create_pos_encoding(self, seq_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, seq_len, input_dim]
            return_attention: Whether to return attention weights from all layers

        Returns:
            output: Predictions based on output_type
            attentions: Optional list of attention weights from each layer
        """
        batch, seq_len, _ = x.shape

        # Validate sequence length - require exact match to avoid silent errors
        # with positional encoding and landmark reshaping
        if seq_len != self.seq_len:
            raise ValueError(
                f"Expected seq_len={self.seq_len}, got {seq_len}. "
                "Pad/trim inputs before calling the model."
            )

        # Project input
        x = self.input_proj(x)

        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]

        # Encode
        attentions = []
        for layer in self.layers:
            x, attn = layer(x, return_attention=return_attention)
            if return_attention:
                attentions.append(attn)

        x = self.norm(x)

        # Use last position for prediction
        x = x[:, -1, :]

        # Output head
        output = self.head(x)

        if return_attention:
            return output, attentions
        return output, None

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss based on output type.

        Args:
            predictions: Model predictions
            targets: Ground truth targets

        Returns:
            loss: Computed loss value
        """
        if self.output_type == 'regression':
            return F.mse_loss(predictions, targets)
        elif self.output_type == 'classification':
            return F.cross_entropy(predictions, targets)
        elif self.output_type == 'allocation':
            # Sharpe-like loss: maximize risk-adjusted returns
            portfolio_returns = predictions * targets
            mean_return = portfolio_returns.mean()
            std_return = portfolio_returns.std() + 1e-8
            return -mean_return / std_return
        else:
            raise ValueError(f"Unknown output_type: {self.output_type}")


def create_nystromformer(
    config: dict = None,
    pretrained: bool = False
) -> NystromformerTrading:
    """
    Factory function to create Nyströmformer model.

    Args:
        config: Model configuration dictionary
        pretrained: Whether to load pretrained weights (not implemented)

    Returns:
        model: NystromformerTrading instance
    """
    default_config = {
        'input_dim': 6,
        'd_model': 256,
        'n_heads': 8,
        'n_layers': 4,
        'num_landmarks': 64,
        'seq_len': 4096,
        'output_type': 'regression',
        'pred_horizon': 24,
        'dropout': 0.1
    }

    if config is not None:
        default_config.update(config)

    model = NystromformerTrading(**default_config)

    if pretrained:
        raise NotImplementedError("Pretrained weights not available yet")

    return model


if __name__ == "__main__":
    # Example usage and quick test
    print("Testing NystromformerTrading...")

    # Create model
    model = NystromformerTrading(
        input_dim=6,
        d_model=128,
        n_heads=4,
        n_layers=2,
        num_landmarks=32,
        seq_len=512,
        output_type='regression',
        pred_horizon=24
    )

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Test forward pass
    batch_size = 4
    seq_len = 512
    input_dim = 6

    x = torch.randn(batch_size, seq_len, input_dim)
    output, _ = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("Test passed!")
