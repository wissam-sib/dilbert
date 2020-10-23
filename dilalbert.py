# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Non interaction size, values priority:
 1 - Class parameter at instantiation
 2 - Config file
 3 - Script default value
"""


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import (
    CrossEntropyLoss,
    MSELoss
)
from transformers.modeling_bert import (
    BertPreTrainedModel,
    BertModel,
    BertLayer
)

from transformers.modeling_albert import (
    AlbertModel,
    AlbertPreTrainedModel
)


from time import perf_counter
from copy import deepcopy

TRAINING = True
EVAL_TIME = False  ## Time Benchmarking
UPDATE_UMBEDDINGS = False
UPDATE_CLS = False  ## Update CLS between part A and part B with token averaging
UPDATE_FINAL_CLS = False  ## Update final CLS with token averaging
IGNORE_CLS_PART_A = False  ## Disable attention mask for cls token in part A and use initial CLS embedding in part B
DEFAULT_NON_INTERACTION_LAYERS = 10
MAX_SEQ_LENGTH = 384  # default value 384
SAME_WEIGHTS_GROUP_A_AND_B = False #Not implemented yet


class DilAlbert(AlbertPreTrainedModel):
    def __init__(self, config, non_interaction_layers=None):
        super(DilAlbert, self).__init__(config)
        self.num_labels = config.num_labels

        self.albert = AlbertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        
        ## Non Interaction Size
        if non_interaction_layers is not None:
            self.non_interaction_layers = non_interaction_layers
            print(f"Dilalbert: non_interaction_layers: Use of class parameter during initialization. Value {self.non_interaction_layers}")
        elif hasattr(config, 'non_interaction_layers'):
            self.non_interaction_layers = config.non_interaction_layers
            print(f"Dilalbert: non_interaction_layers: Use of config file variable. Value {self.non_interaction_layers}")
        else:
            self.non_interaction_layers = DEFAULT_NON_INTERACTION_LAYERS
            print(f"Dilalbert: non_interaction_layers: Use of default value. Value {self.non_interaction_layers}")
        
        if EVAL_TIME:
            self.count = 0
            self.time_perf = {
                'qst tokens count': 0,
                'ctxt tokens count': 0,
                'split qst ctxt': 0,
                'qst process bert input': 0,
                'ctxt process bert input': 0,
                'qst embed': 0,
                'ctxt embed': 0,
                'qst part A': 0,
                'ctxt part A': 0,
                'process bert input': 0,
                'part B': 0,
                'part C': 0
            }

        self.init_weights()
        
        #if SAME_LAYER_GROUP_A_AND_B:
            #copy.deepcopy(model) copy layer group

    
    def split_question_context(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None
    ):
        batch_size = len(input_ids)
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        question_inputs = {'input_ids': [], 'attention_mask': [], 'token_type_ids': [], 'position_ids': [], 'inputs_embeds': []}
        context_inputs = {'input_ids': [], 'attention_mask': [], 'token_type_ids': [], 'position_ids': [], 'inputs_embeds': []}
        
        seq_len = len(input_ids[0])
        split_idxs = []
        for i in range(batch_size):  ## for every example in batch
            idx = (input_ids[i] == 3).nonzero()[0][0] ## Here replace 3 with tokenizer.sep_token_id
            split_idxs.append(idx)
        max_qst_len = max(split_idxs)+1
        max_ctxt_len = len(input_ids[0]) - min(split_idxs) -1
        
        paddings = []
        for i in range(batch_size):  ## for every example in batch
            split_idx = split_idxs[i]
            qst_padding = max_qst_len - split_idx - 1
            ctxt_padding = max_ctxt_len-(seq_len-split_idx-1)
            paddings.append(qst_padding)
            
            if input_ids is not None:
                question_inputs['input_ids'].append(input_ids[i][:split_idx+1].tolist() + [0]*qst_padding)
                context_inputs['input_ids'].append(input_ids[i][split_idx+1:].tolist() + [0]*ctxt_padding)
            if attention_mask is not None:
                question_inputs['attention_mask'].append(attention_mask[i][:split_idx+1].tolist() + [0]*qst_padding)
                context_inputs['attention_mask'].append(attention_mask[i][split_idx+1:].tolist() + [0]*ctxt_padding)
            if token_type_ids is not None:
                question_inputs['token_type_ids'].append(token_type_ids[i][:split_idx+1].tolist() + [0]*qst_padding)
                context_inputs['token_type_ids'].append(token_type_ids[i][split_idx+1:].tolist() + [1]*ctxt_padding)
            ## these embeddings are disabled and the BERT encoder uses the default ones
            if False:
                if position_ids is not None:
                    question_inputs['position_ids'].append(position_ids[i][:split_idx+1].tolist())
                    context_inputs['position_ids'].append(position_ids[i][split_idx+1:].tolist())
                if inputs_embeds is not None:
                    question_inputs['inputs_embeds'].append(inputs_embeds[i][:split_idx+1].tolist())
                    context_inputs['inputs_embeds'].append(inputs_embeds[i][split_idx+1:].tolist())

        question_inputs['input_ids'] = torch.tensor(question_inputs['input_ids'], device=device)#, requires_grad=True)
        question_inputs['token_type_ids'] = torch.tensor(question_inputs['token_type_ids'], device=device)
        question_inputs['position_ids'] = None # torch.tensor(question_inputs['position_ids'], device=device)
        question_inputs['attention_mask'] = torch.tensor(question_inputs['attention_mask'], device=device)
        question_inputs['inputs_embeds'] = None
        context_inputs['input_ids'] = torch.tensor(context_inputs['input_ids'], device=device)#, requires_grad=True)
        context_inputs['token_type_ids'] = torch.tensor(context_inputs['token_type_ids'], device=device)
        context_inputs['position_ids'] = None # torch.tensor(context_inputs['position_ids'], device=device)
        context_inputs['attention_mask'] = torch.tensor(context_inputs['attention_mask'], device=device)
        context_inputs['inputs_embeds'] = None

        return question_inputs, context_inputs, split_idxs, paddings
    
    
    def process_albert_input(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None
    ):
        
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.albert.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.albert.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers
        
        return {
            'input_ids': input_ids,
            'attention_mask': extended_attention_mask,
            'position_ids': position_ids,
            'token_type_ids': token_type_ids,
            'head_mask': head_mask,
            'inputs_embeds': inputs_embeds
        }

    def embed(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, **kwargs):
        """Return BERT embeddings."""
        return self.albert.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )


    def forward_encoder(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        first_part=True
    ):
        
        if first_part:
            hidden_states = self.albert.encoder.embedding_hidden_mapping_in(hidden_states)
            num_layers = self.non_interaction_layers
        else:
            num_layers = self.albert.encoder.config.num_hidden_layers - self.non_interaction_layers
        
        
        all_attentions = ()

        if self.albert.encoder.output_hidden_states:
            all_hidden_states = (hidden_states,)
      
        for i in range(num_layers):                        
            
            # Number of layers in a hidden group
            layers_per_group = int(self.albert.encoder.config.num_hidden_layers / self.albert.encoder.config.num_hidden_groups)

            # Index of the hidden group
            group_idx = int(i / (self.albert.encoder.config.num_hidden_layers / self.albert.encoder.config.num_hidden_groups))
            
            layer_group_output = self.forward_albert_layer_group(group_idx, hidden_states, attention_mask=attention_mask, head_mask=head_mask[group_idx * layers_per_group : (group_idx + 1) * layers_per_group],first_part=first_part)
            
            hidden_states = layer_group_output[0]

            if self.albert.encoder.output_attentions:
                all_attentions = all_attentions + layer_group_output[-1]

            if self.albert.encoder.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.albert.encoder.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.albert.encoder.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)        
    
    def forward_albert_layer_group(self,group_idx, hidden_states, attention_mask=None, head_mask=None,first_part=True):
        
        
        # Not modified in the end, because its not supposed to be here... 
        
        layer_hidden_states = ()
        layer_attentions = ()
        

        for layer_index, albert_layer in enumerate(self.albert.encoder.albert_layer_groups[group_idx].albert_layers):               
            
            layer_output = albert_layer(hidden_states, attention_mask, head_mask[layer_index])
            hidden_states = layer_output[0]

            if self.albert.encoder.albert_layer_groups[group_idx].output_attentions:
                layer_attentions = layer_attentions + (layer_output[1],)

            if self.albert.encoder.albert_layer_groups[group_idx].output_hidden_states:
                layer_hidden_states = layer_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.albert.encoder.albert_layer_groups[group_idx].output_hidden_states:
            outputs = outputs + (layer_hidden_states,)
        if self.albert.encoder.albert_layer_groups[group_idx].output_attentions:
            outputs = outputs + (layer_attentions,)
        return outputs  # last-layer hidden state, (layer hidden states), (layer attentions)
        
    
    def forward_albert(
        self,
        embeddings,
        attention_mask=None,
        head_mask=None,
        first_part=True,
        **kwargs
    ):
        encoder_outputs = self.forward_encoder(
            embeddings,
            attention_mask=attention_mask,
            head_mask=head_mask,
            first_part=first_part
        )
        
        sequence_output = encoder_outputs[0]

        pooled_output = self.albert.pooler_activation(self.albert.pooler(sequence_output[:, 0]))

        outputs = (sequence_output, pooled_output) + encoder_outputs[
            1:
        ]  # add hidden_states and attentions if they are here
        return outputs
        
    
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None
    ):
        torch.set_printoptions(profile="full")
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        t0, t1, t2, t3, t4, t5, t6, t7 = [None]*8
        
        
        if EVAL_TIME: t0 = perf_counter()
        question_inputs, context_inputs, split_idxs, paddings = self.split_question_context(input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, start_positions, end_positions)
        if EVAL_TIME: self.time_perf['split qst ctxt'] += (perf_counter() - t0)

        
        if EVAL_TIME: t1 = perf_counter()
        question_inputs = self.process_albert_input(**question_inputs)
        if EVAL_TIME: t2 = perf_counter()
        embeddings = self.embed(**question_inputs)
        
        
        if IGNORE_CLS_PART_A:
            cls_copy = embeddings[:,0]

        if EVAL_TIME: t3 = perf_counter()

        qst_out = self.forward_albert(
            **question_inputs,
            embeddings=embeddings,
            first_part=True
        )
        
        if EVAL_TIME: t4 = perf_counter()
        context_inputs = self.process_albert_input(**context_inputs)
        if EVAL_TIME: t5 = perf_counter()
        embeddings = self.embed(**context_inputs)
        if EVAL_TIME: t6 = perf_counter()
        ctxt_out = self.forward_albert(
            **context_inputs,
            embeddings=embeddings,
            first_part=True
        )
        
        if EVAL_TIME:
            t7 = perf_counter()
            self.time_perf['qst process bert input'] += (t2-t1)
            self.time_perf['qst embed'] += (t3-t2)
            self.time_perf['qst part A'] += (t4-t3)
            self.time_perf['ctxt process bert input'] += (t5-t4)
            self.time_perf['ctxt embed'] += (t6-t5)
            self.time_perf['ctxt part A'] += (t7-t6)
            t1 = perf_counter()
        
        
        outs = torch.cat((qst_out[0], ctxt_out[0]), 1)
        clone = outs.clone()
        for i, (idx, pad) in enumerate(zip(split_idxs, paddings)):
            if pad == 0: continue
            outs[i, idx+1:-pad] = clone[i, idx+1+pad:]
        hidden_states = outs = outs[:,:MAX_SEQ_LENGTH]
        #print(MAX_SEQ_LENGTH)
        #print(hidden_states.shape)
        #attention_mask = attention_mask[:, :383]
        
        
        if IGNORE_CLS_PART_A:
            hidden_states[:,0] = cls_copy
        
        albert_input = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'position_ids': position_ids,
            'head_mask': head_mask,
            'inputs_embeds':inputs_embeds
        }
        
        albert_input = self.process_albert_input(**albert_input)
        #bert_input['attention_mask'] = torch.cat((question_inputs['attention_mask'], context_inputs['attention_mask']), 3)[:, :, :, :MAX_SEQ_LENGTH]
        
        if UPDATE_CLS:
            hidden_states[:,0] = hidden_states[:,1:].mean(dim=1)
        if UPDATE_UMBEDDINGS:
            input_shape = input_ids.size()
            seq_length = input_shape[1]
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            if position_ids is None:
                position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
                position_ids = position_ids.unsqueeze(0).expand(input_shape)
            hidden_states = (hidden_states +
                            self.bert.embeddings.position_embeddings(position_ids) +
                            self.bert.embeddings.token_type_embeddings(token_type_ids))
            hidden_states = self.bert.embeddings.LayerNorm(hidden_states)
            hidden_states = self.bert.embeddings.dropout(hidden_states)

        if EVAL_TIME: t2 = perf_counter()
        #print(hidden_states.shape)
        #input()
        outputs = self.forward_albert(
            **albert_input,
            embeddings=hidden_states,
            first_part=False
        )
        hidden_states = outputs[0]
        
        if UPDATE_FINAL_CLS:
            hidden_states[:,0] = hidden_states[:,1:].mean(dim=1)
        
        
        if EVAL_TIME: t3 = perf_counter()
        logits = self.qa_outputs(hidden_states)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        if EVAL_TIME:
            t4 = perf_counter()
            self.time_perf['process bert input'] += (t2-t1)
            self.time_perf['part B'] += (t3-t2)
            self.time_perf['part C'] += (t4-t3)

            self.count += 1
            ## print every 50 batches
            if self.count%50 == 0 or self.count == 1036:  # 12430
                for k, v in self.time_perf.items(): print(f"{k}: {v}")
        
        if not TRAINING and False:
            start_logits = start_logits.tolist()
            end_logits = end_logits.tolist()
            fin = MAX_SEQ_LENGTH - max(paddings)
            for i in range(len(start_logits)):
                start_logits[i] = start_logits[i][paddings[i]:paddings[i]+fin]
                end_logits[i] = end_logits[i][paddings[i]:paddings[i]+fin]
            start_logits = torch.tensor(start_logits) #, device=device)
            end_logits = torch.tensor(end_logits) #, device=device)
        
        
        outputs = (start_logits, end_logits,) + outputs[2:] #+ (paddings,)
        if start_positions is not None and end_positions is not None:
            #paddings = torch.tensor(paddings, device=device)
            #start_positions += paddings
            #end_positions += paddings
            
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs


        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)


    #non interaction layers
    def process_A(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None
    ):
        torch.set_printoptions(profile="full")
        albert_input = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'position_ids': position_ids,
            'head_mask': head_mask,
            'inputs_embeds':inputs_embeds
        }
        
        albert_input = self.process_albert_input(**albert_input)
        embeddings = self.embed(**albert_input)
        
        albert_out = self.forward_albert(
            **albert_input,
            embeddings=embeddings,
            first_part=True
        )
        
        return albert_out[0]
    
    # interaction layers
    def process_B(
        self,
        qst_embeddings,
        ctxt_embeddings,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None
    ):
        
        outs = torch.cat((qst_embeddings, ctxt_embeddings), 1)
        hidden_states = outs = outs[:,:MAX_SEQ_LENGTH]

        albert_input = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'position_ids': position_ids,
            'head_mask': head_mask,
            'inputs_embeds':inputs_embeds
        }
        
        albert_input = self.process_albert_input(**albert_input)
        
        outputs = self.forward_albert(
            **albert_input,
            embeddings=hidden_states,
            first_part=False
        )
        hidden_states = outputs[0]
        
        
        logits = self.qa_outputs(hidden_states)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
              
        
        outputs = (start_logits, end_logits,) + outputs[2:] #+ (paddings,)
        if start_positions is not None and end_positions is not None:
            #paddings = torch.tensor(paddings, device=device)
            #start_positions += paddings
            #end_positions += paddings
            
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)
        