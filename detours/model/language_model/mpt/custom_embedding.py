# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class SharedEmbedding(nn.Embedding):

    def forward(self, input: Tensor, unembed: bool=False) -> Tensor:
        if unembed:
            return F.linear(input, self.weight)
        return super().forward(input)