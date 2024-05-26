# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleLocalizationHead(nn.Module):
    def __init__(self, input_dim, num_timestamps):
        super(SimpleLocalizationHead, self).__init__()
        self.fc_start = nn.Linear(input_dim, num_timestamps)
        self.fc_end = nn.Linear(input_dim, num_timestamps)

    def forward(self, x):
        # Assuming x is the output from the transformer with shape [batch_size, num_timestamps, features]
        start_logits = self.fc_start(x)
        end_logits = self.fc_end(x)

        # Apply softmax over the num_timestamps dimension to get a distribution
        start_probs = F.softmax(start_logits, dim=2)
        end_probs = F.softmax(end_logits, dim=2)

        return start_probs, end_probs

class LinearLocalizationHead(nn.Module):
    def __init__(self, input_dim, num_timestamps):
        super(LinearLocalizationHead, self).__init__()
        # Apply a linear transformation to each timestamp feature vector,
        # outputting one logit per timestamp for start and end respectively.
        self.fc_start = nn.Linear(input_dim, 1)
        self.fc_end = nn.Linear(input_dim, 1)

    def forward(self, x, mask=None):
        # x has shape [batch_size, num_timestamps, input_dim]
        # We will process each timestamp separately

        # Process start logits
        start_logits = self.fc_start(x).squeeze(-1)  # Shape: [batch_size, num_timestamps]

        # Process end logits
        end_logits = self.fc_end(x).squeeze(-1)  # Shape: [batch_size, num_timestamps, 1]

        if mask is not None:
            # mask should be a tensor of shape [batch_size, num_timestamps], with 1s for unmasked and 0s for masked
            start_logits = start_logits.masked_fill(mask == 0, float('-inf'))
            end_logits = end_logits.masked_fill(mask == 0, float('-inf'))

        return start_logits, end_logits
