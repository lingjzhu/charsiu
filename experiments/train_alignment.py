#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import os
import argparse
import transformers
import soundfile as sf
import librosa
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from itertools import chain,groupby

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Tuple
from g2p_en import G2p
from datasets import load_dataset, load_from_disk
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Trainer,TrainingArguments
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2ForCTC
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices,Wav2Vec2ForPreTrainingOutput
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import CausalLMOutput, MaskedLMOutput
from transformers import BertForMaskedLM
from transformers import AdamW


class ForwardSumLoss(torch.nn.Module):
    def __init__(self, blank_logprob=-1):
        super(ForwardSumLoss, self).__init__()
        self.log_softmax = torch.nn.LogSoftmax(dim=3)
        self.blank_logprob = blank_logprob
#        self.off_diag_penalty = off_diag_penalty
        self.CTCLoss = nn.CTCLoss(zero_infinity=True)
        
    def forward(self, attn_logprob, text_lens, mel_lens):
        """
        Args:
        attn_logprob: batch x 1 x max(mel_lens) x max(text_lens)
        batched tensor of attention log
        probabilities, padded to length
        of longest sequence in each dimension
        text_lens: batch-D vector of length of
        each text sequence
        mel_lens: batch-D vector of length of
        each mel sequence
        """
        # The CTC loss module assumes the existence of a blank token
        # that can be optionally inserted anywhere in the sequence for
        # a fixed probability.
        # A row must be added to the attention matrix to account for this
        attn_logprob_pd = F.pad(input=attn_logprob,
                                pad=(1, 0, 0, 0, 0, 0, 0, 0),
                                value=self.blank_logprob)
        cost_total = 0.0
        # for-loop over batch because of variable-length
        # sequences
        for bid in range(attn_logprob.shape[0]):
            # construct the target sequence. Every
            # text token is mapped to a unique sequence number,
            # thereby ensuring the monotonicity constraint
            target_seq = torch.arange(1, text_lens[bid]+1)
            target_seq=target_seq.unsqueeze(0)
            curr_logprob = attn_logprob_pd[bid].permute(1, 0, 2)
            curr_logprob = curr_logprob[:mel_lens[bid],:,:text_lens[bid]+1]
            
#            curr_logprob = curr_logprob + self.off_diagonal_loss(curr_logprob,text_lens[bid]+1,mel_lens[bid])
            curr_logprob = self.log_softmax(curr_logprob[None])[0]
            cost = self.CTCLoss(curr_logprob,
                                target_seq,
                                input_lengths=mel_lens[bid:bid+1],
                                target_lengths=text_lens[bid:bid+1])
            cost_total += cost
        # average cost over batch
        cost_total = cost_total/attn_logprob.shape[0]
        return cost_total
    
    def off_diagonal_loss(self,log_prob,N, T, g=0.2):

        n = torch.arange(N).to(log_prob.device)
        t = torch.arange(T).to(log_prob.device)
        t = t.unsqueeze(1).repeat(1,N)
        n = n.unsqueeze(0).repeat(T,1)
    

#        W = 1 - torch.exp(-(n/N - t/T)**2/(2*g**2))
        
#        penalty = log_prob*W.unsqueeze(1)
#        return torch.mean(penalty)
        W = torch.exp(-(n/N - t/T)**2/(2*g**2))
    
        return torch.log_softmax(W.unsqueeze(1),dim=-1)
    
    
class ConvBank(nn.Module):
    def __init__(self, input_dim, output_class_num, kernels, cnn_size, hidden_size, dropout, **kwargs):
        super(ConvBank, self).__init__()
        self.drop_p = dropout
        
        self.in_linear = nn.Linear(input_dim, hidden_size)
        latest_size = hidden_size

        # conv bank
        self.cnns = nn.ModuleList()
        assert len(kernels) > 0
        for kernel in kernels:
            self.cnns.append(nn.Conv1d(latest_size, cnn_size, kernel, padding=kernel//2))
        latest_size = cnn_size * len(kernels)

        self.out_linear = nn.Linear(latest_size, output_class_num)

    def forward(self, features):
        hidden = F.dropout(F.relu(self.in_linear(features)), p=self.drop_p)

        conv_feats = []
        hidden = hidden.transpose(1, 2).contiguous()
        for cnn in self.cnns:   
            conv_feats.append(cnn(hidden))
        hidden = torch.cat(conv_feats, dim=1).transpose(1, 2).contiguous()
        hidden = F.dropout(F.relu(hidden), p=self.drop_p)

        predicted = self.out_linear(hidden)
        return predicted
    
    
class Wav2Vec2ForAlignment(Wav2Vec2ForCTC):
    
    def __init__(self,config):
        super().__init__(config)
        self.cnn = ConvBank(config.hidden_size,384,[1,3],384,384,0.1)
        self.align_loss = ForwardSumLoss()
        
    def freeze_wav2vec2(self):
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
        
    def initialize_phone_model(self,path):
        
        self.bert = BertForMaskedPhoneLM.from_pretrained(path)
        
    
    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        mask_time_indices=None,
        return_dict=None,
        labels=None,
        labels_attention_mask=None,
        text_len=None,
        frame_len=None
    ):
    


        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            mask_time_indices=mask_time_indices,
            return_dict=return_dict,
        )

        # acoustic embeddings
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.cnn(hidden_states)
        
        # phone embeddings
        phone_hidden = self.bert(input_ids=labels,attention_mask=labels_attention_mask).hidden_states[-1]
        
        # compute cross attention
        att = torch.bmm(hidden_states,phone_hidden.transpose(2,1))
        attention_mask = (1-labels_attention_mask)*-10000.0
        att = torch.log_softmax(att+attention_mask.unsqueeze(1).repeat(1,att.size(1),1),dim=-1)
       

        loss = None
        if self.training:
            loss = self.align_loss(att.unsqueeze(1),text_len,frame_len)


        return CausalLMOutput(
            loss=loss, logits=att, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
    

class BertForMaskedPhoneLM(BertForMaskedLM):
    
    def __init__(self,config):
        super().__init__(config)
        self.cnn = ConvBank(config.hidden_size,384,[1],384,384,0.1)
        
    def freeze_feature_extractor(self):
        for param in self.bert.parameters():
            param.requires_grad = False
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=True,
    ):
        

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        
        prediction_scores = self.cnn(outputs.hidden_states[-1])

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



def get_phones(sen):
    '''
    convert texts to phone sequence
    '''
    sen = g2p(sen)
    sen = [re.sub(r'\d','',p) for p in sen]
    return sen

def get_phone_ids(phones):
    '''
    convert phone sequence to ids
    '''
    ids = []
    punctuation = set('.,!?')
    for p in phones:
        if re.match(r'^\w+?$',p):
            ids.append(mapping_phone2id.get(p,mapping_phone2id['[UNK]']))
        elif p in punctuation:
            ids.append(mapping_phone2id.get('[SIL]'))
    ids = [0]+ids
    if ids[-1]!=0:
        ids.append(0)
    return ids #append silence token at the beginning

def audio_preprocess(path):
    
#    features,_ = sf.read(path)
    features, _ = librosa.core.load(path,sr=16000)
    return processor(features, sampling_rate=16000).input_values.squeeze()



def prepare_common_voice_dataset(batch):
    # check that all files have the correct sampling rate

    batch["input_values"] = re.search(r'(.*?)\.mp3', batch['path']).group(1)+'.wav'
    batch['labels'] = batch['sentence']
    return batch


@dataclass
class SpeechCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    return_attention_mask: Optional[bool] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = 256
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        
        # get phone features
        label_features = [{"input_ids": get_phone_ids(get_phones(feature["labels"]))[:self.max_length_labels]} for feature in features]
        text_len = [len(i['input_ids']) for i in label_features] 
        
        with self.processor.as_target_processor():
            label_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_attention_mask=self.return_attention_mask,
                return_tensors="pt",
            )    
        
        # get speech features
        input_features = [{"input_values": audio_preprocess(feature["input_values"])} for feature in features]
        mel_len = [model._get_feat_extract_output_lengths(len(i['input_values'])) for i in input_features]
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_attention_mask=self.return_attention_mask,
            return_tensors="pt",
        )
        batch_size, raw_sequence_length = batch['input_values'].shape
        sequence_length = model._get_feat_extract_output_lengths(raw_sequence_length)
        batch['mask_time_indices'] = _compute_mask_indices((batch_size, sequence_length), mask_prob=0.075, mask_length=2,device='cpu')
        batch['frame_len'] = torch.tensor(mel_len)

        batch['labels'] = label_batch['input_ids']
        batch['text_len'] = torch.tensor(text_len)
        batch['labels_attention_mask'] = label_batch['attention_mask']
        return batch    


if __name__ == "__main__":

    output_dir = '/gpfs/accounts/lingjzhu_root/lingjzhu1/lingjzhu/asr/model/neural_aligner_common_voice_10ms'

    device = 'cuda'


    '''
    Load tokenizers and processors
    '''
    g2p = G2p()
    mapping_phone2id = json.load(open("./vocab.json",'r'))
    mapping_id2phone = {v:k for k,v in mapping_phone2id.items()}

    tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    '''
    Load dataset
    '''
    common_voice = load_from_disk('/gpfs/accounts/lingjzhu_root/lingjzhu1/lingjzhu/asr/common_voice_filtered')
    common_voice = common_voice.map(prepare_common_voice_dataset, remove_columns=common_voice.column_names)
    print(len(common_voice))
    print(common_voice[0])

    speech_collator = SpeechCollatorWithPadding(processor=processor)


    '''
    Load model
    '''
    resolution = 10

    model = Wav2Vec2ForAlignment.from_pretrained('./models/alignment_initialized_weights')
    model.initialize_phone_model('./models/bert-phones/checkpoint-36000')
    weights = torch.load('./models/alignment_initial_weights')
    model.load_state_dict(weights)
    
    if resolution == 10:
        model.wav2vec2.feature_extractor.conv_layers[6].conv.stride = (1,)
        model.config.conv_stride[-1] = 1
        print('The resolution is %s'%resolution)
    model.freeze_feature_extractor()

    '''
    Training loop
    '''
    training_args = TrainingArguments(
                                      output_dir=output_dir,
                                      group_by_length=True,
                                      per_device_train_batch_size=4,
                                      gradient_accumulation_steps=16,
#                                      evaluation_strategy="steps",
                                      num_train_epochs=2,
                                      fp16=True,
                                      save_steps=500,
#                                      eval_steps=1000,
                                      logging_steps=100,
                                      learning_rate=5e-4,
                                      weight_decay=1e-6,
                                      warmup_steps=1000,
                                      save_total_limit=2,
                                      ignore_data_skip=True,
                                     )
        
        
    trainer = Trainer(
                        model=model,
                        data_collator=speech_collator,
                        args=training_args,
#                            compute_metrics=compute_metrics,
                        train_dataset=common_voice,
#                            eval_dataset=libris_train_prepared,
                        tokenizer=processor.feature_extractor,
                        )
    
    
    trainer.train()
