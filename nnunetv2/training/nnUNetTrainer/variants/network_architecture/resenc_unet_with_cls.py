"""
Multi-task network: nnU-Net ResEnc-M segmentation U-Net + a subtype classification head that shares
the encoder.

The classification head is a cross-attention pooling module, ported from the 2nd place solution of the
RSNA 2025 Intracranial Aneurysm Detection challenge
(https://github.com/PengchengShi1220/RSNA2025_Intracranial-Aneurysm-Detection, Apache-2.0), which the
quiz brief points at with the note "please try cross attention pooling".

Difference from the reference implementation: the reference re-implements the whole ResEnc U-Net so it
can attach the head. We instead *wrap* the network that nnU-Net builds from the plans. That keeps the
architecture fully plans-driven, so the mandated nnUNetResEncUNetMPlans still dictates stages, feature
widths, kernels and strides, and the wrapper stays valid if the plans change.

Why cross-attention pooling rather than global average pooling: the subtype signal lives in the lesion,
which occupies a median of 0.5% of the ROI. Global average pooling over the bottleneck would average
that signal into ~96 tokens of mostly-background, whereas learned query tokens can attend to the few
tokens that carry lesion evidence. GAP is kept available behind use_cross_attention=False as a baseline.
"""

from typing import List, Type, Union

import torch
import torch.nn as nn


class CrossAttentionPooling(nn.Module):
    """
    Pools a 3D feature map into class logits with learned query tokens.

    A set of `query_num` learned queries cross-attends over the flattened spatial positions of the
    encoder bottleneck. The attended queries are concatenated (not averaged) so that different queries
    can specialise on different evidence, then projected to logits.

    Shapes:
        input   [B, C, X, Y, Z]  (or [B, C, N])
        tokens  [N, B, C] with N = X*Y*Z
        query   [query_num, B, C]
        output  [B, num_classes]
    """

    def __init__(self, embed_dim: int, query_num: int, num_classes: int, num_heads: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f'embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})')
        self.embed_dim = embed_dim
        self.query_num = query_num
        self.num_classes = num_classes

        self.class_query = nn.Parameter(torch.randn(query_num, embed_dim))
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads,
                                                     dropout=dropout, batch_first=False)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(query_num * embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.class_query)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        if x.dim() == 5:
            x = x.flatten(2)                      # [B, C, N]
        x = x.permute(2, 0, 1)                    # [N, B, C]

        query = self.class_query.unsqueeze(1).expand(-1, batch_size, -1)  # [query_num, B, C]
        attended, _ = self.cross_attention(query=query, key=x, value=x)   # [query_num, B, C]

        attended = self.dropout(self.norm(attended))
        attended = attended.permute(1, 0, 2).flatten(1)  # [B, query_num * C]
        return self.classifier(attended)


class GlobalAveragePooling(nn.Module):
    """Baseline pooling: global average pool -> dropout -> linear."""

    def __init__(self, embed_dim: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self.pool(x).flatten(1)))


class ClassificationHead(nn.Module):
    """Selects between cross-attention pooling and global average pooling."""

    def __init__(self, embed_dim: int, num_classes: int, query_num: int = 4, dropout: float = 0.0,
                 use_cross_attention: bool = True, num_heads: int = 4):
        super().__init__()
        if use_cross_attention:
            self.pooling: nn.Module = CrossAttentionPooling(embed_dim, query_num, num_classes,
                                                            num_heads=num_heads, dropout=dropout)
        else:
            self.pooling = GlobalAveragePooling(embed_dim, num_classes, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pooling(x)


class ResEncUNetWithClassification(nn.Module):
    """
    Wraps a plans-built nnU-Net (must expose `.encoder` returning skips and `.decoder`) and adds a
    classification head on the deepest encoder feature map.

    forward() returns (segmentation_output, classification_logits) where segmentation_output is
    whatever the wrapped decoder returns (a list of tensors under deep supervision, otherwise a single
    tensor), so downstream nnU-Net code that consumes the segmentation output is unchanged.
    """

    def __init__(self, backbone: nn.Module, num_subtypes: int = 3, query_num: int = 4,
                 num_heads: int = 4, cls_dropout: float = 0.0, use_cross_attention: bool = True):
        super().__init__()
        if not hasattr(backbone, 'encoder') or not hasattr(backbone, 'decoder'):
            raise ValueError('backbone must expose .encoder and .decoder (nnU-Net UNet-style network)')
        if not getattr(backbone.encoder, 'return_skips', False):
            raise ValueError('backbone.encoder must be built with return_skips=True')

        self.backbone = backbone
        self.num_subtypes = num_subtypes
        embed_dim = int(backbone.encoder.output_channels[-1])
        self.classification_head = ClassificationHead(
            embed_dim=embed_dim,
            num_classes=num_subtypes,
            query_num=query_num,
            dropout=cls_dropout,
            use_cross_attention=use_cross_attention,
            num_heads=num_heads,
        )

    # nnU-Net toggles this attribute on the network when switching deep supervision on/off
    @property
    def decoder(self):
        return self.backbone.decoder

    @property
    def encoder(self):
        return self.backbone.encoder

    def forward(self, x: torch.Tensor):
        skips = self.backbone.encoder(x)
        if getattr(self, 'capture_classification_embedding', False):
            bottleneck = skips[-1].float()
            spatial_dims = tuple(range(2, bottleneck.ndim))
            self.last_classification_embedding = torch.cat(
                (bottleneck.mean(dim=spatial_dims), bottleneck.amax(dim=spatial_dims)), dim=1
            ).detach()
        segmentation = self.backbone.decoder(skips)
        logits = self.classification_head(skips[-1])
        return segmentation, logits

    @torch.no_grad()
    def forward_classification_only(self, x: torch.Tensor) -> torch.Tensor:
        """Encoder + classification head only. Skips the decoder entirely."""
        skips = self.backbone.encoder(x)
        return self.classification_head(skips[-1])

    def compute_conv_feature_map_size(self, input_size):
        """Delegate VRAM estimation to the wrapped network (the head's cost is negligible)."""
        return self.backbone.compute_conv_feature_map_size(input_size)
