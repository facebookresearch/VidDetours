# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union, Dict

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.nn.utils.rnn import pad_sequence
import argparse
import numpy as np

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast, SequenceClassifierOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

from ..retrieval_arch import TransformerClassifier, TransformerLocalizer, VideoLocalizationTransformer
from ..localization_arch import VSLNet
from ..localization_simple import LinearLocalizationHead
from ..localization_utils import visual_feature_sampling, pad_video_seq, convert_length_to_mask, pad_seq, time_to_index, index_to_time


class LlavaConfig(LlamaConfig):
    model_type = "llava"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config, task_name='captioning', head_config={}, full_video_retrieval=True, max_duration=600):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)

        self.task_name = task_name
        if self.task_name == 'captioning':
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        elif self.task_name == 'only_retrieval':
            self.retrieval_aggregator = head_config['retrieval_aggregator']
            if self.retrieval_aggregator == 'transformer':
                self.retrieval_head = TransformerClassifier(4096, 4, 4, 1024, 1)
            elif self.retrieval_aggregator == 'average':
                self.retrieval_head = nn.Linear(4096, 1)
        elif self.task_name == 'only_localization':
            head_config.update({'dim': config.hidden_size})
            head_config = argparse.Namespace(**head_config)
            self.localization_head = VSLNet(configs=head_config, word_vectors=None)
            self.localization_n_chunks = head_config.n_chunks
            # self.localization_head = LinearLocalizationHead(5120, self.localization_n_chunks)
            # self.localization_head = VideoLocalizationTransformer(5120, 4, 4, 1024, self.localization_n_chunks)
            self.localization_video_len = head_config.video_len
        self.full_video_retrieval = full_video_retrieval # Whether to use full video features for retrieval (Not used for localization)
        self.max_duration = max_duration # Maximum duration of video features (Only used in full video retrieval)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        videos1: Optional[torch.FloatTensor] = None,
        videos2: Optional[torch.FloatTensor] = None,
        videos2_startends: Optional[torch.FloatTensor] = None,
        videos2_retrieval_labels: Optional[torch.FloatTensor] = None,
        videos2_localization_labels: Optional[Dict[str, torch.FloatTensor]] = None,
        return_dict: Optional[bool] = None,
        video1_ids: Optional[List[int]] = None, # For debugging
        video2_ids: Optional[List[int]] = None, # For debugging
        instance_ids: Optional[int] = None, # For testing
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.task_name in ["only_retrieval", "only_localization"]:
            # Our modification
            '''
            labels is None, use videos2_startends as the label for temporal localization
            '''
            input_ids, attention_mask, past_key_values, inputs_embeds, labels, mm_features_info = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images, videos1, videos2, videos2_startends, use_feats=True)
        else:
            input_ids, attention_mask, past_key_values, inputs_embeds, labels, _ = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        if self.task_name == "captioning":
            logits = self.lm_head(hidden_states)
        elif self.task_name == "only_localization":
            # Prepare lables, The idea is that the batch will contain start and end time in seconds and trainer loop is responsible to do `time_to_index` and `index_to_time`.
            s_labels = []
            e_labels = []
            feature_lenths = []
            for batch_idx in range(len(videos2_startends)):
                assert len(videos2_startends) == len(hidden_states)
                start_sec = videos2_localization_labels['s_labels'][batch_idx].item()
                end_sec = videos2_localization_labels['e_labels'][batch_idx].item()
                vid_duration = mm_features_info[batch_idx]['videos2'][0]['len']
                feature_lenths.append(min(self.localization_n_chunks, vid_duration))
                s_ind, e_ind, _ = time_to_index(start_sec, end_sec, min(self.localization_n_chunks, vid_duration), vid_duration)
                s_labels.append(s_ind)
                e_labels.append(e_ind)
            # Generate h_labels same as VSLNet
            max_length = max(feature_lenths)
            h_labels = np.zeros(shape=[len(videos2_startends), max_length], dtype=np.int16)
            extend = 0.1
            for idx in range(len(videos2_startends)):
                st, et = s_labels[idx].item(), e_labels[idx].item()
                cur_max_len = feature_lenths[idx]
                extend_len = round(extend * float(et - st + 1))
                if extend_len > 0:
                    st_ = max(0, st - extend_len)
                    et_ = min(et + extend_len, cur_max_len - 1)
                    h_labels[idx][st_:(et_ + 1)] = 1
                else:
                    h_labels[idx][st:(et + 1)] = 1
            s_labels = torch.tensor(s_labels).to(device=hidden_states.device)
            e_labels = torch.tensor(e_labels).to(device=hidden_states.device)
            h_labels = torch.tensor(h_labels).to(device=hidden_states.device, dtype=hidden_states.dtype)
            all_video_feats = []
            # hidden_states contains the full sequence, we first extract the video 2 duration and then give it for localization
            for batch_idx, hidden_state in enumerate(hidden_states):
                assert videos2_retrieval_labels[batch_idx] is None or videos2_retrieval_labels[batch_idx] == 1, "Label should be 1, currently it is {}".format(videos2_retrieval_labels[batch_idx]) # For localization, we only consider the correct samples
                # First get the complete video feats and then use the annotation to further trim it
                start_video = mm_features_info[batch_idx]['videos2'][0]['start']
                len_video = mm_features_info[batch_idx]['videos2'][0]['len']
                cur_video_feats = hidden_state[start_video:(start_video + len_video), :]
                assert self.localization_n_chunks != -1, "You need to set the localization_n_chunks before calling forward."
                rescaled_video_feats = visual_feature_sampling(cur_video_feats, self.localization_n_chunks)
                all_video_feats.append(rescaled_video_feats)
            vfeats, vfeat_lens = pad_video_seq(all_video_feats, dtype=hidden_states.dtype)
            vmask = convert_length_to_mask(vfeat_lens).to(device=vfeats.device, dtype=vfeats.dtype)

            # # Debug print
            # # Print original video length and start and end times, and then rescaled start, end and length
            # print('*'*100)
            # print(instance_ids)
            # print(f"Video 1 ids {video1_ids} and video 2 ids {video2_ids}")
            # print(f"Video 2 startends: {videos2_startends}")
            # print(f"Original video length {[mm_features_info[x]['videos2'][0]['len'] for x in range(len(mm_features_info))]}, start time {videos2_localization_labels['s_labels']}, end time {videos2_localization_labels['e_labels']}")
            # print(f"Rescaled video length {vfeat_lens}, start time {s_labels}, end time {e_labels}")
            # exit()

            # GET LANGUAGE EMBEDDINGS
            all_text_feats = []
            all_video1_feats = []
            for batch_idx, hidden_state in enumerate(hidden_states):
                # The text lies between the end of Video 1 and the start of Video 2
                start_text = mm_features_info[batch_idx]['videos1'][0]['start'] + mm_features_info[batch_idx]['videos1'][0]['len'] + 1
                end_text = mm_features_info[batch_idx]['videos2'][0]['start']
                cur_text_feats = hidden_state[start_text:end_text, :]
                all_text_feats.append(cur_text_feats)

                # Also add the video 1
                start_video = mm_features_info[batch_idx]['videos1'][0]['start']
                end_video = mm_features_info[batch_idx]['videos1'][0]['start'] + mm_features_info[batch_idx]['videos1'][0]['len']
                cur_video_feats = hidden_state[start_video:end_video, :]
                all_video1_feats.append(cur_video_feats)

            video_token_length = self.localization_video_len
            text_token_length = self.localization_n_chunks - video_token_length
            tfeats, tfeat_lens = pad_seq(all_text_feats, max_length=text_token_length, padding='maximum')
            tmask = convert_length_to_mask(tfeat_lens, padding='maximum', max_length=text_token_length).to(device=tfeats.device, dtype=tfeats.dtype)

            v1feats, v1feat_lens = pad_seq(all_video1_feats, max_length=video_token_length, trim_from_start=True, padding='maximum')
            v1mask = convert_length_to_mask(v1feat_lens, trim_from_start=True, padding='maximum', max_length=video_token_length).to(device=v1feats.device, dtype=v1feats.dtype)

            # Video 1 is masked at the start and text is masked at the end
            tfeats = torch.cat([v1feats, tfeats], dim=1)
            tmask = torch.cat([v1mask, tmask], dim=1)

            # TODO: ashutoshkr, We can now proceed with copy pasting the model design
            # start_logits, end_logits = self.localization_head(vfeats, vmask)
            # criterion = CrossEntropyLoss()
            # start_loss = criterion(start_logits, s_labels)
            # end_loss = criterion(end_logits, e_labels)
            # loss = start_loss + end_loss
            # # start and end indices for inference
            # start_indices = torch.argmax(torch.softmax(start_logits, dim=1), dim=1)
            # end_indices = torch.argmax(torch.softmax(end_logits, dim=1), dim=1)
            # for batch_idx in range(len(start_logits)):
            #     assert start_indices[batch_idx].item() <= vfeat_lens[batch_idx].item(), f"Start index {start_indices[batch_idx].item()} exceeds the video length {vfeat_lens[batch_idx].item()}"
            #     assert end_indices[batch_idx].item() <= vfeat_lens[batch_idx].item(), f"End index {end_indices[batch_idx].item()} exceeds the video length {vfeat_lens[batch_idx].item()}"
            # print(f"Start indices {start_indices}, End indices {end_indices}")
            # print(f"Start labels {s_labels}, End labels {e_labels}")
            # print('*'*100)
            # BREAK POINT BETWEEN THE TWO METHODS
            h_score, start_logits, end_logits = self.localization_head(None, None, vfeats, vmask, tmask, tfeats)
            highlight_loss = self.localization_head.compute_highlight_loss(h_score, h_labels, vmask)
            loc_loss = self.localization_head.compute_loss(start_logits, end_logits, s_labels, e_labels)
            loss = loc_loss + self.localization_head.get_config().highlight_lambda * highlight_loss

            # The following parts are only for evaluation
            start_indices, end_indices = self.localization_head.extract_index(start_logits.to(torch.float32), end_logits.to(torch.float32))
            # Convert back to original time
            predicted_start_time = []
            predicted_end_time = []
            for batch_idx in range(len(start_indices)):
                vid_duration = mm_features_info[batch_idx]['videos2'][0]['len']
                curr_start_time, curr_end_time = index_to_time(start_indices[batch_idx].item(), end_indices[batch_idx].item(), min(self.localization_n_chunks, vid_duration), vid_duration)
                predicted_start_time.append(curr_start_time)
                predicted_end_time.append(curr_end_time)

            return SequenceClassifierOutput(
                loss=loss,
                # Add feature length and duration here
                logits= {
                    'localization_pred_start_times': torch.tensor(predicted_start_time).to(hidden_states.device) ,
                    'localization_pred_end_times': torch.tensor(predicted_end_time).to(hidden_states.device),
                    'localization_instance_ids': instance_ids}
                )

        elif self.task_name == "only_retrieval":
            # Extract video tokens from the relevant times
            window_hidden_states = []
            for batch_idx, hidden_state in enumerate(hidden_states):
                # First get the complete video feats and then use the annotation to further trim it
                start_video = mm_features_info[batch_idx]['videos2'][0]['start']
                len_video = mm_features_info[batch_idx]['videos2'][0]['len']
                cur_video_feats = hidden_state[start_video:(start_video + len_video), :]
                if self.full_video_retrieval:
                    candidate_feats = cur_video_feats[:self.max_duration]
                else:
                    candidate_feats = cur_video_feats[int(videos2_startends[batch_idx, 0].item()):int(videos2_startends[batch_idx, 1].item())]
                window_hidden_states.append(candidate_feats)

            # Pad the sequences
            seq_lengths = [seq.size(0) for seq in window_hidden_states]
            max_seq_length = max(seq_lengths)
            padded_input_batch = torch.zeros(len(window_hidden_states), max_seq_length, 4096, dtype=hidden_state.dtype).to(hidden_states.device)

            for i, seq in enumerate(window_hidden_states):
                padded_input_batch[i, :seq_lengths[i], :] = seq

            # Create the src_key_padding_mask
            src_key_padding_mask = padded_input_batch.sum(dim=-1) == 0

            if self.retrieval_aggregator == 'transformer':
                preds = self.retrieval_head(padded_input_batch, src_key_padding_mask=src_key_padding_mask)
            elif self.retrieval_aggregator == 'average':
                # 1. Use mask to nullify unwanted tokens.
                masked_data = padded_input_batch * src_key_padding_mask.unsqueeze(-1)  # Add singleton dim to mask and multiply.
                # 2. Sum along sequence length.
                sum_data = torch.sum(masked_data, dim=1)
                # 3. Sum mask to get total weight per batch item.
                sum_mask = torch.sum(src_key_padding_mask, dim=1, keepdim=True)
                # Avoid division by zero: replace 0s in sum_mask with 1s
                sum_mask = torch.where(sum_mask == 0, torch.ones_like(sum_mask), sum_mask)
                # 4. Average.
                average_data = sum_data / sum_mask
                preds = self.retrieval_head(average_data)
            else:
                raise NotImplementedError("Only transformer and average aggregators are supported")

            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(preds, videos2_retrieval_labels.to(dtype=preds.dtype))

            return SequenceClassifierOutput(
                loss=loss,
                logits= {
                    'retrieval_logits': preds,
                    'instance_ids': instance_ids
                }
            )

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

# AutoConfig.register("llava", LlavaConfig)
# AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
