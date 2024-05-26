# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


'''
Layers code taken from: https://github.com/EGO4D/episodic-memory/blob/5275b9570a3d77884e8685bad474309f7778db1f/NLQ/VSLNet/model/layers.py#L1C1-L570C37
with the following changes:
    - Changed the dtype to always match the input dtype
    - Restored single prediction as in VSLNet instead of 5 predictions done by Ego4D NLQ baseline

Other utils have the source written in the function docs.
'''

"""
Layers to construct the VSLNet model.
"""
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def mask_logits(inputs, mask, mask_value=-1e30, dtype=torch.float32):
    mask = mask.type(dtype)
    return inputs + (1.0 - mask) * mask_value


class Conv1D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=True):
        super(Conv1D, self).__init__()
        self.conv1d = nn.Conv1d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=bias,
        )

    def forward(self, x):
        # suppose all the input with shape (batch_size, seq_len, dim)
        x = x.transpose(1, 2)  # (batch_size, dim, seq_len)
        x = self.conv1d(x)
        return x.transpose(1, 2)  # (batch_size, seq_len, dim)


class WordEmbedding(nn.Module):
    def __init__(self, num_words, word_dim, drop_rate, word_vectors=None):
        super(WordEmbedding, self).__init__()
        self.is_pretrained = False if word_vectors is None else True
        if self.is_pretrained:
            self.pad_vec = nn.Parameter(
                torch.zeros(size=(1, word_dim), dtype=torch.float32),
                requires_grad=False,
            )
            unk_vec = torch.empty(
                size=(1, word_dim), requires_grad=True, dtype=torch.float32
            )
            nn.init.xavier_uniform_(unk_vec)
            self.unk_vec = nn.Parameter(unk_vec, requires_grad=True)
            self.glove_vec = nn.Parameter(
                torch.tensor(word_vectors, dtype=torch.float32), requires_grad=False
            )
        else:
            self.word_emb = nn.Embedding(num_words, word_dim, padding_idx=0)
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, word_ids):
        if self.is_pretrained:
            word_emb = F.embedding(
                word_ids,
                torch.cat([self.pad_vec, self.unk_vec, self.glove_vec], dim=0),
                padding_idx=0,
            )
        else:
            word_emb = self.word_emb(word_ids)
        return self.dropout(word_emb)


class CharacterEmbedding(nn.Module):
    def __init__(self, num_chars, char_dim, drop_rate):
        super(CharacterEmbedding, self).__init__()
        self.char_emb = nn.Embedding(num_chars, char_dim, padding_idx=0)
        kernels, channels = [1, 2, 3, 4], [10, 20, 30, 40]
        self.char_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=char_dim,
                        out_channels=channel,
                        kernel_size=(1, kernel),
                        stride=(1, 1),
                        padding=0,
                        bias=True,
                    ),
                    nn.ReLU(),
                )
                for kernel, channel in zip(kernels, channels)
            ]
        )
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, char_ids):
        char_emb = self.char_emb(
            char_ids
        )  # (batch_size, w_seq_len, c_seq_len, char_dim)
        char_emb = self.dropout(char_emb)
        char_emb = char_emb.permute(
            0, 3, 1, 2
        )  # (batch_size, char_dim, w_seq_len, c_seq_len)
        char_outputs = []
        for conv_layer in self.char_convs:
            output = conv_layer(char_emb)
            output, _ = torch.max(
                output, dim=3, keepdim=False
            )  # reduce max (batch_size, channel, w_seq_len)
            char_outputs.append(output)
        char_output = torch.cat(
            char_outputs, dim=1
        )  # (batch_size, sum(channels), w_seq_len)
        return char_output.permute(0, 2, 1)  # (batch_size, w_seq_len, sum(channels))


class Embedding(nn.Module):
    def __init__(
        self,
        num_words,
        num_chars,
        word_dim,
        char_dim,
        drop_rate,
        out_dim,
        word_vectors=None,
    ):
        super(Embedding, self).__init__()
        self.word_emb = WordEmbedding(
            num_words, word_dim, drop_rate, word_vectors=word_vectors
        )
        self.char_emb = CharacterEmbedding(num_chars, char_dim, drop_rate)
        # output linear layer
        self.linear = Conv1D(
            in_dim=word_dim + 100,
            out_dim=out_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(self, word_ids, char_ids):
        word_emb = self.word_emb(word_ids)  # (batch_size, w_seq_len, word_dim)
        char_emb = self.char_emb(char_ids)  # (batch_size, w_seq_len, 100)
        emb = torch.cat(
            [word_emb, char_emb], dim=2
        )  # (batch_size, w_seq_len, word_dim + 100)
        emb = self.linear(emb)  # (batch_size, w_seq_len, dim)
        return emb


class BertEmbedding(nn.Module):
    def __init__(self, text_agnostic=False):
        super(BertEmbedding, self).__init__()
        from transformers import BertModel, BertConfig

        # Load pretrained BERT model if not text_agnostic, else random BERT.
        if text_agnostic:
            self.embedder = BertModel(BertConfig())
        else:
            self.embedder = BertModel.from_pretrained("bert-base-uncased")
        # Freeze the model.
        for param in self.embedder.parameters():
            param.requires_grad = False

    def forward(self, word_ids):
        outputs = self.embedder(**word_ids)
        return outputs["last_hidden_state"].detach()


class PositionalEmbedding(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, num_embeddings, embedding_dim):
        super(PositionalEmbedding, self).__init__()
        self.position_embeddings = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, inputs):
        bsz, seq_length = inputs.shape[:2]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=inputs.device)
        position_ids = position_ids.unsqueeze(0).repeat(bsz, 1)  # (N, L)
        position_embeddings = self.position_embeddings(position_ids)
        return position_embeddings


class VisualProjection(nn.Module):
    def __init__(self, visual_dim, dim, drop_rate=0.0):
        super(VisualProjection, self).__init__()
        self.drop = nn.Dropout(p=drop_rate)
        self.linear = Conv1D(
            in_dim=visual_dim,
            out_dim=dim,
            kernel_size=1,
            stride=1,
            bias=True,
            padding=0,
        )

    def forward(self, visual_features):
        # the input visual feature with shape (batch_size, seq_len, visual_dim)
        visual_features = self.drop(visual_features)
        output = self.linear(visual_features)  # (batch_size, seq_len, dim)
        return output


class DepthwiseSeparableConvBlock(nn.Module):
    def __init__(self, dim, kernel_size, drop_rate, num_layers=4):
        super(DepthwiseSeparableConvBlock, self).__init__()
        self.depthwise_separable_conv = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=dim,
                        out_channels=dim,
                        kernel_size=kernel_size,
                        groups=dim,
                        padding=kernel_size // 2,
                        bias=False,
                    ),
                    nn.Conv1d(
                        in_channels=dim,
                        out_channels=dim,
                        kernel_size=1,
                        padding=0,
                        bias=True,
                    ),
                    nn.ReLU(),
                )
                for _ in range(num_layers)
            ]
        )
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(dim, eps=1e-6) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, x):
        output = x  # (batch_size, seq_len, dim)
        for idx, conv_layer in enumerate(self.depthwise_separable_conv):
            residual = output
            output = self.layer_norms[idx](output)  # (batch_size, seq_len, dim)
            output = output.transpose(1, 2)  # (batch_size, dim, seq_len)
            output = conv_layer(output)
            output = self.dropout(output)
            output = output.transpose(1, 2) + residual  # (batch_size, seq_len, dim)
        return output


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, drop_rate):
        super(MultiHeadAttentionBlock, self).__init__()
        assert (
            dim % num_heads == 0
        ), "The channels (%d) is not a multiple of attention heads (%d)" % (
            dim,
            num_heads,
        )
        self.head_size, self.num_heads, self.dim = int(dim / num_heads), num_heads, dim
        self.dropout = nn.Dropout(p=drop_rate)
        self.query = Conv1D(
            in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.key = Conv1D(
            in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.value = Conv1D(
            in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.layer_norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.out_layer = Conv1D(
            in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True
        )

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, w_seq_len, head_size)

    @staticmethod
    def combine_last_two_dim(x):
        old_shape = list(x.size())
        new_shape = old_shape[:-2] + [old_shape[-2] * old_shape[-1]]
        return x.reshape(shape=new_shape)

    def forward(self, x, mask=None):
        output = self.layer_norm1(x)  # (batch_size, seq_len, dim)
        output = self.dropout(output)
        # multi-head attention layer
        query = self.transpose_for_scores(
            self.query(output)
        )  # (batch_size, num_heads, seq_len, head_size)
        key = self.transpose_for_scores(self.key(output))
        value = self.transpose_for_scores(self.value(output))
        attention_scores = torch.matmul(
            query, key.transpose(-1, -2)
        )  # (batch_size, num_heads, seq_len, seq_len)
        attention_scores = attention_scores / math.sqrt(self.head_size)
        if mask is not None:  # masking
            mask = mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
            attention_scores = mask_logits(attention_scores, mask, dtype=attention_scores.dtype)
        attention_probs = nn.Softmax(dim=-1)(
            attention_scores
        )  # (batch_size, num_heads, seq_len, seq_len)
        attention_probs = self.dropout(attention_probs)
        value = torch.matmul(
            attention_probs, value
        )  # (batch_size, num_heads, seq_len, head_size)
        value = self.combine_last_two_dim(
            value.permute(0, 2, 1, 3)
        )  # (batch_size, seq_len, dim)
        # intermediate layer
        output = self.dropout(value)
        residual = output + x
        output = self.layer_norm2(residual)
        output = self.dropout(output)
        output = self.out_layer(output)
        output = self.dropout(output) + residual
        return output


class FeatureEncoder(nn.Module):
    def __init__(
        self, dim, num_heads, max_pos_len, kernel_size=7, num_layers=4, drop_rate=0.0
    ):
        super(FeatureEncoder, self).__init__()
        self.pos_embedding = PositionalEmbedding(
            num_embeddings=max_pos_len, embedding_dim=dim
        )
        self.conv_block = DepthwiseSeparableConvBlock(
            dim=dim, kernel_size=kernel_size, drop_rate=drop_rate, num_layers=num_layers
        )
        self.attention_block = MultiHeadAttentionBlock(
            dim=dim, num_heads=num_heads, drop_rate=drop_rate
        )

    def forward(self, x, mask=None):
        features = x + self.pos_embedding(x)  # (batch_size, seq_len, dim)
        features = self.conv_block(features)  # (batch_size, seq_len, dim)
        features = self.attention_block(
            features, mask=mask
        )  # (batch_size, seq_len, dim)
        return features


class CQAttention(nn.Module):
    def __init__(self, dim, drop_rate=0.0):
        super(CQAttention, self).__init__()
        w4C = torch.empty(dim, 1)
        w4Q = torch.empty(dim, 1)
        w4mlu = torch.empty(1, 1, dim)
        nn.init.xavier_uniform_(w4C)
        nn.init.xavier_uniform_(w4Q)
        nn.init.xavier_uniform_(w4mlu)
        self.w4C = nn.Parameter(w4C, requires_grad=True)
        self.w4Q = nn.Parameter(w4Q, requires_grad=True)
        self.w4mlu = nn.Parameter(w4mlu, requires_grad=True)
        self.dropout = nn.Dropout(p=drop_rate)
        self.cqa_linear = Conv1D(
            in_dim=4 * dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True
        )

    def forward(self, context, query, c_mask, q_mask):
        score = self.trilinear_attention(
            context, query
        )  # (batch_size, c_seq_len, q_seq_len)
        score_ = nn.Softmax(dim=2)(
            mask_logits(score, q_mask.unsqueeze(1), dtype=score.dtype)
        )  # (batch_size, c_seq_len, q_seq_len)
        score_t = nn.Softmax(dim=1)(
            mask_logits(score, c_mask.unsqueeze(2), dtype=score.dtype)
        )  # (batch_size, c_seq_len, q_seq_len)
        score_t = score_t.transpose(1, 2)  # (batch_size, q_seq_len, c_seq_len)
        c2q = torch.matmul(score_, query)  # (batch_size, c_seq_len, dim)
        q2c = torch.matmul(
            torch.matmul(score_, score_t), context
        )  # (batch_size, c_seq_len, dim)
        output = torch.cat(
            [context, c2q, torch.mul(context, c2q), torch.mul(context, q2c)], dim=2
        )
        output = self.cqa_linear(output)  # (batch_size, c_seq_len, dim)
        return output

    def trilinear_attention(self, context, query):
        batch_size, c_seq_len, dim = context.shape
        batch_size, q_seq_len, dim = query.shape
        context = self.dropout(context)
        query = self.dropout(query)
        subres0 = torch.matmul(context, self.w4C).expand(
            [-1, -1, q_seq_len]
        )  # (batch_size, c_seq_len, q_seq_len)
        subres1 = (
            torch.matmul(query, self.w4Q).transpose(1, 2).expand([-1, c_seq_len, -1])
        )
        subres2 = torch.matmul(context * self.w4mlu, query.transpose(1, 2))
        res = subres0 + subres1 + subres2  # (batch_size, c_seq_len, q_seq_len)
        return res


class WeightedPool(nn.Module):
    def __init__(self, dim):
        super(WeightedPool, self).__init__()
        weight = torch.empty(dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def forward(self, x, mask):
        alpha = torch.tensordot(
            x, self.weight, dims=1
        )  # shape = (batch_size, seq_length, 1)
        alpha = mask_logits(alpha, mask=mask.unsqueeze(2), dtype=alpha.dtype)
        alphas = nn.Softmax(dim=1)(alpha)
        pooled_x = torch.matmul(x.transpose(1, 2), alphas)  # (batch_size, dim, 1)
        pooled_x = pooled_x.squeeze(2)
        return pooled_x


class CQConcatenate(nn.Module):
    def __init__(self, dim):
        super(CQConcatenate, self).__init__()
        self.weighted_pool = WeightedPool(dim=dim)
        self.conv1d = Conv1D(
            in_dim=2 * dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True
        )

    def forward(self, context, query, q_mask):
        pooled_query = self.weighted_pool(query, q_mask)  # (batch_size, dim)
        _, c_seq_len, _ = context.shape
        pooled_query = pooled_query.unsqueeze(1).repeat(
            1, c_seq_len, 1
        )  # (batch_size, c_seq_len, dim)
        output = torch.cat(
            [context, pooled_query], dim=2
        )  # (batch_size, c_seq_len, 2*dim)
        output = self.conv1d(output)
        return output


class HighLightLayer(nn.Module):
    def __init__(self, dim):
        super(HighLightLayer, self).__init__()
        self.conv1d = Conv1D(
            in_dim=dim, out_dim=1, kernel_size=1, stride=1, padding=0, bias=True
        )

    def forward(self, x, mask):
        # compute logits
        logits = self.conv1d(x)
        logits = logits.squeeze(2)
        logits = mask_logits(logits, mask, dtype=logits.dtype)
        # compute score
        scores = nn.Sigmoid()(logits)
        return scores

    @staticmethod
    def compute_loss(scores, labels, mask, epsilon=1e-12):
        weights = torch.where(labels == 0.0, labels + 1.0, 2.0 * labels)
        loss_per_location = nn.BCELoss(reduction="none")(scores, labels)
        loss_per_location = loss_per_location * weights
        loss = torch.sum(loss_per_location * mask) / (torch.sum(mask) + epsilon)
        return loss


class DynamicRNN(nn.Module):
    def __init__(self, dim):
        super(DynamicRNN, self).__init__()
        self.lstm = nn.LSTM(
            input_size=dim,
            hidden_size=dim,
            num_layers=1,
            bias=True,
            batch_first=True,
            bidirectional=False,
        )

    def forward(self, x, mask):
        out, _ = self.lstm(x)  # (bsz, seq_len, dim)
        mask = mask.type(torch.float32)
        mask = mask.unsqueeze(2)
        out = out * mask
        return out


class ConditionedPredictor(nn.Module):
    def __init__(self, dim, num_heads, max_pos_len, drop_rate=0.0, predictor="rnn"):
        super(ConditionedPredictor, self).__init__()
        self.predictor = predictor
        if predictor == "rnn":
            self.start_encoder = DynamicRNN(dim=dim)
            self.end_encoder = DynamicRNN(dim=dim)
        else:
            self.encoder = FeatureEncoder(
                dim=dim,
                num_heads=num_heads,
                kernel_size=7,
                num_layers=4,
                max_pos_len=max_pos_len,
                drop_rate=drop_rate,
            )
            self.start_layer_norm = nn.LayerNorm(dim, eps=1e-6)
            self.end_layer_norm = nn.LayerNorm(dim, eps=1e-6)

        self.start_block = nn.Sequential(
            Conv1D(
                in_dim=2 * dim,
                out_dim=dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.ReLU(),
            Conv1D(
                in_dim=dim, out_dim=1, kernel_size=1, stride=1, padding=0, bias=True
            ),
        )
        self.end_block = nn.Sequential(
            Conv1D(
                in_dim=2 * dim,
                out_dim=dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.ReLU(),
            Conv1D(
                in_dim=dim, out_dim=1, kernel_size=1, stride=1, padding=0, bias=True
            ),
        )

    def forward(self, x, mask):
        if self.predictor == "rnn":
            start_features = self.start_encoder(x, mask)  # (batch_size, seq_len, dim)
            end_features = self.end_encoder(start_features, mask)
        else:
            start_features = self.encoder(x, mask)
            end_features = self.encoder(start_features, mask)
            start_features = self.start_layer_norm(start_features)
            end_features = self.end_layer_norm(end_features)
        start_features = self.start_block(
            torch.cat([start_features, x], dim=2)
        )  # (batch_size, seq_len, 1)
        end_features = self.end_block(torch.cat([end_features, x], dim=2))
        start_logits = mask_logits(start_features.squeeze(2), mask=mask, dtype=start_features.dtype)
        end_logits = mask_logits(end_features.squeeze(2), mask=mask, dtype=end_features.dtype)
        return start_logits, end_logits

    @staticmethod
    def extract_index(start_logits, end_logits):
        start_prob = nn.Softmax(dim=1)(start_logits)
        end_prob = nn.Softmax(dim=1)(end_logits)
        outer = torch.matmul(start_prob.unsqueeze(dim=2), end_prob.unsqueeze(dim=1))
        outer = torch.triu(outer, diagonal=0)

        # Restore the original VSLNet implementation of having only one prediction
        _, start_index = torch.max(torch.max(outer, dim=2)[0], dim=1)  # (batch_size, )
        _, end_index = torch.max(torch.max(outer, dim=1)[0], dim=1)  # (batch_size, )
        return start_index, end_index

        # Get top 5 start and end indices.
        batch_size, height, width = outer.shape
        outer_flat = outer.view(batch_size, -1)
        _, flat_indices = outer_flat.topk(5, dim=-1)
        start_indices = flat_indices // width
        end_indices = flat_indices % width
        return start_indices, end_indices

    @staticmethod
    def compute_cross_entropy_loss(start_logits, end_logits, start_labels, end_labels):
        start_loss = nn.CrossEntropyLoss(reduction="mean")(start_logits, start_labels)
        end_loss = nn.CrossEntropyLoss(reduction="mean")(end_logits, end_labels)
        return start_loss + end_loss

# Below code is taken from other points in the VSLNet repo. All sources in function docs.
def visual_feature_sampling(visual_feature, max_num_clips):
    '''
    Converted to PyTorch from https://github.com/EGO4D/episodic-memory/blob/5275b9570a3d77884e8685bad474309f7778db1f/NLQ/VSLNet/utils/data_util.py#L66
    '''
    device = visual_feature.device  # Get the device of the input tensor
    num_clips = visual_feature.shape[0]

    if num_clips <= max_num_clips:
        return visual_feature

    # Create the index tensor
    idxs = torch.arange(0, max_num_clips + 1, 1.0, device=device) / max_num_clips * num_clips
    idxs = torch.round(idxs).to(torch.int32)
    idxs[idxs > num_clips - 1] = num_clips - 1

    # Initialize an empty list to hold the new visual features
    new_visual_feature = []

    for i in range(max_num_clips):
        s_idx, e_idx = idxs[i], idxs[i + 1]
        if s_idx < e_idx:
            # Note: torch.mean supports autograd
            new_visual_feature.append(torch.mean(visual_feature[s_idx:e_idx], dim=0))
        else:
            new_visual_feature.append(visual_feature[s_idx])

    # Convert list of tensors to a single tensor
    new_visual_feature = torch.stack(new_visual_feature)

    return new_visual_feature


def pad_video_seq(sequences, max_length=None, dtype=torch.float32):
    '''
    Converted to PyTorch from https://github.com/EGO4D/episodic-memory/blob/5275b9570a3d77884e8685bad474309f7778db1f/NLQ/VSLNet/utils/data_util.py#L166C1-L180C44
    Additional changes:
    - Added dtype argument to change dtype as per the input
    '''
    if max_length is None:
        max_length = max([seq.shape[0] for seq in sequences])

    feature_length = sequences[0].shape[1]
    device = sequences[0].device  # Get the device of the first tensor in the list
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        # Make sure the sequence is on the same device as the first tensor in the list
        assert seq.device == device, "All tensors must be on the same device."

        add_length = max_length - seq.shape[0]
        sequence_length.append(seq.shape[0])

        if add_length > 0:
            # Create a zero tensor for padding
            add_feature = torch.zeros(add_length, feature_length, device=device, dtype=dtype)

            # Concatenate the sequence with the padding tensor
            seq_ = torch.cat([seq, add_feature], dim=0)
        else:
            seq_ = seq

        sequence_padded.append(seq_)

    # Convert list of sequence lengths to a tensor
    sequence_length = torch.tensor(sequence_length, device=device, dtype=dtype)

    # Convert list of padded sequences to a tensor
    sequence_padded = torch.stack(sequence_padded)

    return sequence_padded, sequence_length

def pad_seq(sequences, pad_tok=None, max_length=None, trim_from_start=False, padding='minimum'):
    if pad_tok is None:
        pad_tok = 0  # 0: "PAD"
    if max_length is None:
        max_length = max([len(seq) for seq in sequences])
    else:
        if padding == 'minimum':
            # When text tokens are smaller, we want the max_length to behave as if max_length=None
            max_length = min(max_length, max([len(seq) for seq in sequences]))
        else:
            # Return the full size
            max_length = max_length

    feature_length = sequences[0].shape[1]
    device = sequences[0].device  # Get the device of the first tensor in the list
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        # Make sure the sequence is on the same device as the first tensor in the list
        assert seq.device == device, "All tensors must be on the same device."

        cur_length = len(seq)
        add_length = max(max_length - cur_length, 0)

        if add_length > 0:
            pad_tensor = torch.full((add_length, feature_length), pad_tok, device=device, dtype=seq.dtype)
            seq_ = torch.cat([seq, pad_tensor])
        else:
            if trim_from_start:
                start_index = max(0, cur_length - max_length)
                seq_ = seq[start_index:min(cur_length, start_index + max_length)]
            else:
                seq_ = seq[:min(cur_length, max_length)]

        sequence_padded.append(seq_)
        sequence_length.append(min(cur_length, max_length))

    # Convert list of sequence lengths to a tensor
    sequence_length = torch.tensor(sequence_length, device=device, dtype=torch.int32)

    # Convert list of padded sequences to a tensor
    sequence_padded = torch.stack(sequence_padded)

    return sequence_padded, sequence_length

def convert_length_to_mask(lengths, trim_from_start=False, padding='minimum', max_length=None):
    if padding == 'minimum':
        max_len = int(lengths.max().item())
    else:
        assert max_length is not None
        max_len = max_length
    mask = torch.arange(max_len, device=lengths.device).expand(lengths.size()[0], max_len)# < lengths.unsqueeze(1)
    if trim_from_start:
        mask = mask >= (max_len - lengths.unsqueeze(1))
    else:
        mask = mask < lengths.unsqueeze(1)
    mask = mask.float()
    return mask

def compute_overlap(pred, gt):
    # check format
    assert isinstance(pred, list) and isinstance(gt, list)
    pred_is_list = isinstance(pred[0], list)
    gt_is_list = isinstance(gt[0], list)
    pred = pred if pred_is_list else [pred]
    gt = gt if gt_is_list else [gt]
    # compute overlap
    pred, gt = np.array(pred), np.array(gt)
    inter_left = np.maximum(pred[:, 0, None], gt[None, :, 0])
    inter_right = np.minimum(pred[:, 1, None], gt[None, :, 1])
    inter = np.maximum(0.0, inter_right - inter_left)
    union_left = np.minimum(pred[:, 0, None], gt[None, :, 0])
    union_right = np.maximum(pred[:, 1, None], gt[None, :, 1])
    union = np.maximum(1e-12, union_right - union_left)
    overlap = 1.0 * inter / union
    # reformat output
    overlap = overlap if gt_is_list else overlap[:, 0]
    overlap = overlap if pred_is_list else overlap[0]
    return overlap

def time_to_index(start_time, end_time, num_units, duration):
    s_times = np.arange(0, num_units).astype(np.float32) / float(num_units) * duration
    e_times = np.arange(1, num_units + 1).astype(np.float32) / float(num_units) * duration
    candidates = np.stack([np.repeat(s_times[:, None], repeats=num_units, axis=1),
                           np.repeat(e_times[None, :], repeats=num_units, axis=0)], axis=2).reshape((-1, 2))
    overlaps = compute_overlap(candidates.tolist(), [start_time, end_time]).reshape(num_units, num_units)
    start_index = np.argmax(overlaps) // num_units
    end_index = np.argmax(overlaps) % num_units
    return start_index, end_index, overlaps

def index_to_time(start_index, end_index, num_units, duration):
    s_times = np.arange(0, num_units).astype(np.float32) * duration / float(num_units)
    e_times = np.arange(1, num_units + 1).astype(np.float32) * duration / float(num_units)
    start_time = s_times[start_index]
    end_time = e_times[end_index]
    return start_time, end_time
