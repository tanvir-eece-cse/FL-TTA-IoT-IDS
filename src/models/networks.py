"""
Neural Network Models for IoT IDS
- MLP Baseline
- TabTransformer (FT-Transformer style)
- With BatchNorm for TTA compatibility
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import math


class MLP(nn.Module):
    """
    MLP with BatchNorm layers for TTA (Test-Time Adaptation) compatibility.
    BatchNorm statistics can be adapted at test time.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [512, 256, 128],
        num_classes: int = 2,
        dropout: float = 0.2,
        use_batchnorm: bool = True,
        activation: str = 'relu'
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.use_batchnorm = use_batchnorm
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            if activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'silu':
                layers.append(nn.SiLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        return self.classifier(features)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get intermediate features before classifier."""
        return self.features(x)


class TabularEmbedding(nn.Module):
    """Embedding layer for tabular data (continuous features)."""
    
    def __init__(self, input_dim: int, embed_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        # Linear projection for each feature
        self.projections = nn.ModuleList([
            nn.Linear(1, embed_dim) for _ in range(input_dim)
        ])
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, input_dim)
        embeddings = []
        for i, proj in enumerate(self.projections):
            feat = x[:, i:i+1]  # (batch, 1)
            embeddings.append(proj(feat))  # (batch, embed_dim)
        
        # Stack: (batch, input_dim, embed_dim)
        embedded = torch.stack(embeddings, dim=1)
        return self.layer_norm(embedded)


class TransformerBlock(nn.Module):
    """Transformer encoder block."""
    
    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x


class FTTransformer(nn.Module):
    """
    FT-Transformer for tabular data.
    Based on: "Revisiting Deep Learning Models for Tabular Data"
    
    Uses per-feature embeddings + transformer encoder + CLS token for classification.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        ff_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # Feature embedding
        self.embedding = TabularEmbedding(input_dim, embed_dim)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Positional encoding (learnable)
        self.pos_embedding = nn.Parameter(torch.randn(1, input_dim + 1, embed_dim))
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        # Embed features: (batch, input_dim, embed_dim)
        embedded = self.embedding(x)
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embedded = torch.cat([cls_tokens, embedded], dim=1)
        
        # Add positional encoding
        embedded = embedded + self.pos_embedding
        
        # Transformer blocks
        for block in self.transformer_blocks:
            embedded = block(embedded)
        
        # Use CLS token for classification
        cls_output = embedded[:, 0]
        return self.classifier(cls_output)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get CLS token features before classifier."""
        batch_size = x.shape[0]
        embedded = self.embedding(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embedded = torch.cat([cls_tokens, embedded], dim=1)
        embedded = embedded + self.pos_embedding
        
        for block in self.transformer_blocks:
            embedded = block(embedded)
        
        return embedded[:, 0]


class TabNetEncoder(nn.Module):
    """Simplified TabNet-style attention encoder."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_steps: int = 3,
        relaxation_factor: float = 1.5,
        sparsity_coefficient: float = 1e-4
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        self.relaxation_factor = relaxation_factor
        self.sparsity_coefficient = sparsity_coefficient
        
        # Shared layers
        self.initial_bn = nn.BatchNorm1d(input_dim)
        
        # Feature transformer for each step
        self.feature_transformers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True)
            ) for _ in range(num_steps)
        ])
        
        # Attention transformers
        self.attention_transformers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, input_dim),
                nn.BatchNorm1d(input_dim)
            ) for _ in range(num_steps)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial_bn(x)
        
        # Prior scales (for sparsity)
        prior_scales = torch.ones_like(x)
        aggregated = torch.zeros(x.shape[0], self.hidden_dim, device=x.device)
        
        for step in range(self.num_steps):
            # Feature transformation
            h = self.feature_transformers[step](x)
            aggregated = aggregated + h
            
            # Attention (for feature selection)
            if step < self.num_steps - 1:
                attn = self.attention_transformers[step](h)
                attn = attn * prior_scales
                attn = F.softmax(attn, dim=-1)
                prior_scales = prior_scales * (self.relaxation_factor - attn)
                x = x * attn
        
        return aggregated


class IoTIDSModel(nn.Module):
    """
    Unified IoT IDS Model supporting multiple architectures.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        architecture: str = 'mlp',  # 'mlp', 'ft_transformer', 'tabnet'
        **kwargs
    ):
        super().__init__()
        self.architecture = architecture
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        if architecture == 'mlp':
            hidden_dims = kwargs.get('hidden_dims', [512, 256, 128])
            dropout = kwargs.get('dropout', 0.2)
            self.model = MLP(input_dim, hidden_dims, num_classes, dropout)
        
        elif architecture == 'ft_transformer':
            embed_dim = kwargs.get('embed_dim', 64)
            num_heads = kwargs.get('num_heads', 4)
            num_layers = kwargs.get('num_layers', 3)
            self.model = FTTransformer(
                input_dim, num_classes, embed_dim, num_heads, num_layers
            )
        
        elif architecture == 'tabnet':
            hidden_dim = kwargs.get('hidden_dim', 256)
            num_steps = kwargs.get('num_steps', 3)
            self.encoder = TabNetEncoder(input_dim, hidden_dim, num_steps)
            self.classifier = nn.Linear(hidden_dim, num_classes)
        
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.architecture == 'tabnet':
            features = self.encoder(x)
            return self.classifier(features)
        return self.model(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.architecture == 'tabnet':
            return self.encoder(x)
        return self.model.get_features(x)


def create_model(
    input_dim: int,
    num_classes: int = 2,
    architecture: str = 'mlp',
    **kwargs
) -> nn.Module:
    """Factory function to create models."""
    return IoTIDSModel(input_dim, num_classes, architecture, **kwargs)
