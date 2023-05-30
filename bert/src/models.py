import logging
import math
import os
import sys

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import *
import numpy as np
import copy
import pickle


class SpellBert(BertPreTrainedModel):
    def __init__(self, config):
        super(SpellBert, self).__init__(config)

        self.vocab_size = config.vocab_size
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.vocab_size)

        self.init_weights()

    def tie_cls_weight(self):
        self.classifier.weight = self.bert.embeddings.word_embeddings.weight

    @staticmethod
    def build_batch(batch, tokenizer):
        return batch

    def forward(self, batch):
        input_ids = batch['src_idx']
        attention_mask = batch['masks']
        loss_mask = batch['loss_masks']
        label_ids = batch['tgt_idx'] if 'tgt_idx' in batch else None

        outputs = self.bert(input_ids, attention_mask=attention_mask)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if label_ids is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            active_loss = loss_mask.view(-1) == 1
            active_logits = logits.view(-1, self.vocab_size)[active_loss]
            active_labels = label_ids.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
            outputs = (loss,) + outputs

            if not self.training:
                new_input_ids = copy.deepcopy(input_ids)

                pred_label = active_logits.argmax(-1)

                new_input_ids.view(-1)[active_loss] = pred_label

                new_outputs = self.bert(new_input_ids, attention_mask=attention_mask)

                new_sequence_output = new_outputs[0]

                new_sequence_output = self.dropout(new_sequence_output)

                new_logits = self.classifier(new_sequence_output)

                outputs = (outputs[0], (new_logits + logits) / 2)

        return outputs


class SRFBert(BertPreTrainedModel):
    def __init__(self, config):
        super(SRFBert, self).__init__(config)

        self.vocab_size = config.vocab_size
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.vocab_size)

        self.init_weights()

    def tie_cls_weight(self):
        self.classifier.weight = self.bert.embeddings.word_embeddings.weight

    @staticmethod
    def build_batch(batch, tokenizer):
        return batch

    def forward(self, batch):
        input_ids = batch['src_idx']
        attention_mask = batch['masks']
        loss_mask = batch['loss_masks']
        label_ids = batch['tgt_idx'] if 'tgt_idx' in batch else None
        bsz, max_length = input_ids.shape

        outputs = self.bert(input_ids, attention_mask=attention_mask)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)

        logits = self.classifier(sequence_output)
        last_hidden_state = outputs.last_hidden_state

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if label_ids is not None:
            # Only keep active parts of the CrossEntropy loss
            loss_fct = CrossEntropyLoss()
            active_loss = loss_mask.view(-1) == 1
            active_logits = logits.view(-1, self.vocab_size)[active_loss]
            active_labels = label_ids.view(-1)[active_loss]
            loss1 = loss_fct(active_logits, active_labels)

            if self.training:

                # Contrastive Probability ptimization Loss
                wrong_judge_positions = ~torch.eq(input_ids, label_ids)

                # self refine
                new_input_ids = copy.deepcopy(input_ids)
                # new_input_ids=copy.deepcopy(label_ids)

                pred_label = active_logits.argmax(-1)

                wrong_judge_positions = ~torch.eq(input_ids, new_input_ids)

                new_input_ids.view(-1)[active_loss] = pred_label
                if new_input_ids.view(-1)[wrong_judge_positions.view(-1)].size(0):
                    new_input_ids.view(-1)[wrong_judge_positions.view(-1)] = logits.view(-1, self.vocab_size)[
                        wrong_judge_positions.view(-1)].argmax(-1)

                # no_mask_position=pred_label!=103
                # new_input_ids.view(-1)[active_loss][no_mask_position]=pred_label[no_mask_position]

                new_outputs = self.bert(new_input_ids, attention_mask=attention_mask)
                #
                new_sequence_output = new_outputs[0]

                new_sequence_output = self.dropout(new_sequence_output)

                new_logits = self.classifier(new_sequence_output)

                loss_fct = CrossEntropyLoss()
                new_active_logits = new_logits.view(-1, self.vocab_size)[active_loss]
                loss3 = loss_fct(new_active_logits, active_labels)
                #
                p = torch.log_softmax(new_active_logits, dim=-1)
                p_tec = torch.softmax(new_active_logits, dim=-1)
                q = torch.log_softmax(active_logits, dim=-1)
                q_tec = torch.softmax(active_logits, dim=-1)

                kl_loss = torch.nn.functional.kl_div(p, q_tec, reduction='none').sum()
                reverse_kl_loss = torch.nn.functional.kl_div(q, p_tec, reduction='none').sum()

                #
                loss = 0.5 * (loss1 + loss3)
                loss += 0.001 * (kl_loss + reverse_kl_loss) / 2

                outputs = (loss,) + outputs


            else:
                outputs = (loss1,) + outputs

                total_logits = logits
                for i in range(2):
                    new_input_ids = copy.deepcopy(input_ids)

                    pred_label = active_logits.argmax(-1)

                    new_input_ids.view(-1)[active_loss] = pred_label

                    new_outputs = self.bert(new_input_ids, attention_mask=attention_mask)

                    new_sequence_output = new_outputs[0]

                    new_sequence_output = self.dropout(new_sequence_output)

                    new_logits = self.classifier(new_sequence_output)

                    total_logits += new_logits

                    input_ids = copy.deepcopy(new_input_ids)
                    active_logits = new_logits.view(-1, self.vocab_size)[active_loss]

                # outputs=(outputs[0],(new_logits+logits)/2)
                outputs = (outputs[0], new_logits)

        return outputs