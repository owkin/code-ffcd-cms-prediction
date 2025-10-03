# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utility functions for attention mechanisms."""

import torch
from typing import Optional

from .tile_layers import MaskedLinear



class GatedAttention(torch.nn.Module):
    """Gated Attention, as defined in https://arxiv.org/abs/1802.04712. Permutation invariant Layer on dim 1.

    Parameters.
    ----------
    d_model: int = 128
    temperature: float = 1.0
        Attention Softmax temperature
    """

    def __init__(
        self,
        d_model: int = 128,
        temperature: float = 1.0,
    ):
        super().__init__()

        self.att = torch.nn.Linear(d_model, d_model)
        self.gate = torch.nn.Linear(d_model, d_model)
        self.w = MaskedLinear(d_model, 1, "-inf")

        self.temperature = temperature

    def attention(
        self,
        v: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """Gets attention logits.

        Parameters.
        ----------
        v: torch.Tensor
            (B, SEQ_LEN, IN_FEATURES)
        mask: Optional[torch.BoolTensor] = None
            (B, SEQ_LEN, 1), True for values that were padded.

        Returns:
        -------
        attention_logits: torch.Tensor
            (B, N_TILES, 1)
        """
        h_v = self.att(v)
        h_v = torch.tanh(h_v)

        u_v = self.gate(v)
        u_v = torch.sigmoid(u_v)

        attention_logits = self.w(h_v * u_v, mask=mask) / self.temperature
        return attention_logits

    def forward(
        self, v: torch.Tensor, mask: Optional[torch.BoolTensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Parameters.
        ----------
        v: torch.Tensor
            (B, SEQ_LEN, IN_FEATURES)
        mask: Optional[torch.BoolTensor] = None
            (B, SEQ_LEN, 1), True for values that were padded.

        Returns:
        -------
        scaled_attention, attention_weights: Tuple[torch.Tensor, torch.Tensor]
            (B, IN_FEATURES), (B, N_TILES, 1)
        """
        attention_logits = self.attention(v=v, mask=mask)

        attention_weights = torch.softmax(attention_logits, 1)
        scaled_attention = torch.matmul(attention_weights.transpose(1, 2), v)

        return scaled_attention.squeeze(1), attention_weights
