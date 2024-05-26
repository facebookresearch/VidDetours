# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


'''
Taken from: https://github.com/EGO4D/episodic-memory/blob/5275b9570a3d77884e8685bad474309f7778db1f/NLQ/VSLNet/model/VSLNet.py#L1C1-L151C10
with the following changes:
    - Added a config option "llm" to skip visual encoder and text network since we get that already from LLM
    - Modifies the forward definition to accept processed query_features instead of word_ids/char_ids
    - Adds a `get_config()` function
'''
import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup

from .localization_utils import (
    Embedding,
    VisualProjection,
    FeatureEncoder,
    CQAttention,
    CQConcatenate,
    ConditionedPredictor,
    HighLightLayer,
    BertEmbedding,
)


def build_optimizer_and_scheduler(model, configs):
    no_decay = [
        "bias",
        "layer_norm",
        "LayerNorm",
    ]  # no decay for parameters of layer norm and bias
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=configs.init_lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        configs.num_train_steps * configs.warmup_proportion,
        configs.num_train_steps,
    )
    return optimizer, scheduler


class VSLNet(nn.Module):
    def __init__(self, configs, word_vectors):
        super(VSLNet, self).__init__()
        self.configs = configs
        if self.configs.predictor != "llm":
            self.video_affine = VisualProjection(
                visual_dim=configs.video_feature_dim,
                dim=configs.dim,
                drop_rate=configs.drop_rate,
            )
        # self.feature_encoder = FeatureEncoder(
        #     dim=configs.dim,
        #     num_heads=configs.num_heads,
        #     kernel_size=7,
        #     num_layers=4,
        #     max_pos_len=configs.max_pos_len,
        #     drop_rate=configs.drop_rate,
        # )
        # video and query fusion
        self.cq_attention = CQAttention(dim=configs.dim, drop_rate=configs.drop_rate)
        self.cq_concat = CQConcatenate(dim=configs.dim)
        # query-guided highlighting
        self.highlight_layer = HighLightLayer(dim=configs.dim)
        # conditioned predictor
        self.predictor = ConditionedPredictor(
            dim=configs.dim,
            num_heads=configs.num_heads,
            drop_rate=configs.drop_rate,
            max_pos_len=configs.max_pos_len,
            predictor=configs.predictor,
        )

        # If pretrained transformer, initialize_parameters and load.
        if configs.predictor == "bert":
            # Project back from BERT to dim.
            self.query_affine = nn.Linear(768, configs.dim)
            # init parameters
            self.init_parameters()
            self.embedding_net = BertEmbedding(configs.text_agnostic)
        elif configs.predictor == "llm":
            pass #, we don't need embedding_net
        else:
            self.embedding_net = Embedding(
                num_words=configs.word_size,
                num_chars=configs.char_size,
                out_dim=configs.dim,
                word_dim=configs.word_dim,
                char_dim=configs.char_dim,
                word_vectors=word_vectors,
                drop_rate=configs.drop_rate,
            )
            # init parameters
            self.init_parameters()

    def init_parameters(self):
        def init_weights(m):
            if (
                isinstance(m, nn.Conv2d)
                or isinstance(m, nn.Conv1d)
                or isinstance(m, nn.Linear)
            ):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                m.reset_parameters()

        self.apply(init_weights)

    def forward(self, word_ids, char_ids, video_features, v_mask, q_mask, query_features=None):
        if self.configs.predictor != "llm":
            video_features = self.video_affine(video_features)
            if self.configs.predictor == "bert":
                query_features = self.embedding_net(word_ids)
                query_features = self.query_affine(query_features)
            else:
                query_features = self.embedding_net(word_ids, char_ids)
        else:
            assert word_ids is None, "word ids must be None when processed is True"
            assert char_ids is None, "char ids must be None when processed is True"
            assert query_features is not None, "query features must be provided when processed is True"
            # When processed, q_mask is 2-dimensional, same as v_mask

        # print('*'*100)
        # print(f"Input shape is {video_features.shape}, {query_features.shape}")
        # query_features = self.feature_encoder(query_features, mask=q_mask)
        # video_features = self.feature_encoder(video_features, mask=v_mask)
        # print(f"Output shape is {video_features.shape}, {query_features.shape}")
        features = self.cq_attention(video_features, query_features, v_mask, q_mask)
        features = self.cq_concat(features, query_features, q_mask)
        h_score = self.highlight_layer(features, v_mask)
        features = features * h_score.unsqueeze(2)
        start_logits, end_logits = self.predictor(features, mask=v_mask)
        return h_score, start_logits, end_logits

    def extract_index(self, start_logits, end_logits):
        return self.predictor.extract_index(
            start_logits=start_logits, end_logits=end_logits
        )

    def compute_highlight_loss(self, scores, labels, mask):
        return self.highlight_layer.compute_loss(
            scores=scores, labels=labels, mask=mask
        )

    def compute_loss(self, start_logits, end_logits, start_labels, end_labels):
        return self.predictor.compute_cross_entropy_loss(
            start_logits=start_logits,
            end_logits=end_logits,
            start_labels=start_labels,
            end_labels=end_labels,
        )

    def get_config(self):
        return self.configs
