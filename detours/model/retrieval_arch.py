# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerClassifier(nn.Module):
    def __init__(self, emb_dim, nhead, num_layers, dim_feedforward, num_classes=1):
        super(TransformerClassifier, self).__init__()
        self.emb_dim = emb_dim
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.emb_dim))

        self.pos_encoder = PositionalEncoding(emb_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(emb_dim, nhead, dim_feedforward),
            num_layers
        )

        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, x, src_key_padding_mask):
        # Add [CLS] token
        cls_tokens = self.cls_token.repeat(x.size(0), 1, 1).to(x.device)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional encodings
        x = self.pos_encoder(x)

        # Add CLS mask
        cls_mask = torch.zeros(src_key_padding_mask.size(0), 1, dtype=torch.bool).to(x.device)
        src_key_padding_mask = torch.cat((cls_mask, src_key_padding_mask), dim=1)

        x = x.permute(1, 0, 2)
        # print('*'*100)
        # print(x.shape, src_key_padding_mask.shape)
        # print(self.transformer_encoder.device, x.device, x.dtype, src_key_padding_mask.device, src_key_padding_mask.dtype)
        # Pass through the transformer
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        # Use the representation of [CLS] token for classification
        cls_output = x[0]
        logit = self.classifier(cls_output)

        return logit

class VideoLocalizationTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, max_seq_len):
        super(VideoLocalizationTransformer, self).__init__()

        # Positional encoding is important for sequence data in Transformers
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)

        # Transformer Encoder Layer
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model,
                                                    nhead=nhead,
                                                    dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)

        # Output layers for start and end logits
        self.start_logits = nn.Linear(d_model, 1)
        self.end_logits = nn.Linear(d_model, 1)

    def forward(self, src, src_mask=None):
        # src shape: [batch_size, seq_len, d_model]

        # Change src to shape [seq_len, batch_size, d_model] to match PyTorch's Transformer input requirements
        src = src.permute(1, 0, 2)

        # Add positional encoding
        src = self.pos_encoder(src)

        # Pass through the Transformer encoder, applying the mask if provided
        transformer_output = self.transformer_encoder(src, src_key_padding_mask=src_mask)

        # Change back to shape [batch_size, seq_len, d_model]
        transformer_output = transformer_output.permute(1, 0, 2)

        # Apply the output layers to get logits for start and end positions
        start_logits = self.start_logits(transformer_output).squeeze(-1)
        end_logits = self.end_logits(transformer_output).squeeze(-1)

        if src_mask is not None:
            # mask should be a tensor of shape [batch_size, num_timestamps], with 1s for unmasked and 0s for masked
            start_logits = start_logits.masked_fill(src_mask == 0, float('-inf'))
            end_logits = end_logits.masked_fill(src_mask == 0, float('-inf'))

        return start_logits, end_logits

class TransformerLocalizer(nn.Module):
    def __init__(self, emb_dim, nhead, num_layers, dim_feedforward, max_seq_length=256):
        super(TransformerLocalizer, self).__init__()
        self.emb_dim = emb_dim
        self.pos_encoder = PositionalEncoding(emb_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(emb_dim, nhead, dim_feedforward),
            num_layers
        )

        # Two separate classifiers for predicting the start and end indices
        self.start_classifier = nn.Linear(emb_dim, max_seq_length)
        self.end_classifier = nn.Linear(emb_dim, max_seq_length)

    def forward(self, x, src_key_padding_mask):
        # Add positional encodings
        print('Input shape is {}'.format(x.shape))
        x = self.pos_encoder(x)

        x = x.permute(1, 0, 2)

        # Pass through the transformer
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Pass the output through the classifiers to predict start and end positions
        start_logits = self.start_classifier(x.permute(1, 0, 2))
        end_logits = self.end_classifier(x.permute(1, 0, 2))
        print('%'*100)
        print(start_logits.shape, end_logits.shape)
        exit()
        return start_logits, end_logits

# ... (The rest of the PositionalEncoding class remains the same)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :].to(x.device)
        return x
