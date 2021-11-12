#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import re
import argparse
import transformers
import soundfile as sf
import librosa
import jiwer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from g2p_en import G2p
import numpy as np
from datasets import load_dataset, load_metric, load_from_disk
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Trainer,TrainingArguments
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2ForPreTraining,Wav2Vec2Config
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices,Wav2Vec2ForPreTrainingOutput



def prepare_common_voice_dataset(batch):
    # check that all files have the correct sampling rate

    batch["input_values"] = re.search(r'(.*?)\.mp3', batch['path']).group(1)+'.wav'
    return batch

def prepare_multicn_dataset(batch):
    # check that all files have the correct sampling rate

    batch["input_values"] = batch['path']
    return batch


def audio_preprocess(path):
    
    features,sr = sf.read(path)
    assert sr == 16000
    return processor(features, sampling_rate=16000).input_values.squeeze()


@dataclass
class DataCollatorWithPadding:
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
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": audio_preprocess(feature["input_values"])} for feature in features]

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
        batch['mask_time_indices'] = _compute_mask_indices((batch_size, sequence_length), mask_prob=0.1, mask_length=2,device='cpu')

        return batch



tokenizer = Wav2Vec2CTCTokenizer("vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token=" ")
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)



if __name__ == "__main__":
    
    lang = 'en'
    # loading  data
    if lang == 'en':
        common_voice = load_from_disk('/gpfs/accounts/lingjzhu_root/lingjzhu1/lingjzhu/asr/common_voice_filtered')
        data_prepared = common_voice.map(prepare_common_voice_dataset, remove_columns=common_voice.column_names)
        print('English data ready!')
    elif lang == 'zh':
        multicn = load_from_disk('/gpfs/accounts/lingjzhu_root/lingjzhu1/lingjzhu/asr/multicn_list')
        data_prepared = multicn.map(prepare_multicn_dataset, remove_columns=multicn.column_names)
        print('Chinese data ready!')
        
 
    
    
    # data loader
    data_collator = DataCollatorWithPadding(processor=processor, padding=True)
    

    # load model
    config = Wav2Vec2Config()
    config.num_attention_heads = 6
    config.hidden_size = 384
    config.num_hidden_layers = 6
    config.num_negatives = 20
    model = Wav2Vec2ForPreTraining(config)
    
    if lang == 'en':
        pre_trained_model = Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-base")
        
        layers = {'wav2vec2.feature_extractor.conv_layers.0.conv.weight', 'wav2vec2.feature_extractor.conv_layers.0.layer_norm.weight', 'wav2vec2.feature_extractor.conv_layers.0.layer_norm.bias', 'wav2vec2.feature_extractor.conv_layers.1.conv.weight', 'wav2vec2.feature_extractor.conv_layers.2.conv.weight', 'wav2vec2.feature_extractor.conv_layers.3.conv.weight', 'wav2vec2.feature_extractor.conv_layers.4.conv.weight', 'wav2vec2.feature_extractor.conv_layers.5.conv.weight', 'wav2vec2.feature_extractor.conv_layers.6.conv.weight','quantizer.codevectors', 'quantizer.weight_proj.weight', 'quantizer.weight_proj.bias', 'project_q.weight', 'project_q.bias'}
        pretrained_dict = {k: v for k, v in  pre_trained_model.state_dict().items() if k in layers}
        print('Loaded Wav2Vec2 English')
    elif lang == 'zh':
        pre_trained_model = Wav2Vec2ForPreTraining.from_pretrained('facebook/wav2vec2-large-xlsr-53')
        
        layers = {'wav2vec2.feature_extractor.conv_layers.0.conv.weight', 'wav2vec2.feature_extractor.conv_layers.0.layer_norm.weight', 'wav2vec2.feature_extractor.conv_layers.0.layer_norm.bias', 'wav2vec2.feature_extractor.conv_layers.1.conv.weight', 'wav2vec2.feature_extractor.conv_layers.2.conv.weight', 'wav2vec2.feature_extractor.conv_layers.3.conv.weight', 'wav2vec2.feature_extractor.conv_layers.4.conv.weight', 'wav2vec2.feature_extractor.conv_layers.5.conv.weight', 'wav2vec2.feature_extractor.conv_layers.6.conv.weight'}
        pretrained_dict = {k:v for k,v in pre_trained_model.state_dict().items() if k in layers}
        print('Loaded xlsr-53 weights')

    
    state_dict = model.state_dict()
    state_dict.update(pretrained_dict)
    model.load_state_dict(state_dict)
    
    model.freeze_feature_extractor()
    model.wav2vec2.feature_extractor.conv_layers[6].conv.stride = (1,)
    model.config.conv_stride[-1] = 1
    
    del pre_trained_model
    
    
    if lang == 'en':
        output_dir = "/scratch/lingjzhu_root/lingjzhu1/lingjzhu/asr/wav2vec2-common_voice-pretraining"
    elif lang == 'zh':
        output_dir = "/scratch/lingjzhu_root/lingjzhu1/lingjzhu/asr/wav2vec2-multicn-pretraining"
    
    
    # training settings
    training_args = TrainingArguments(
                                      output_dir=output_dir,
                                      group_by_length=True,
                                      per_device_train_batch_size=4,
                                      gradient_accumulation_steps=40,
#                                      evaluation_strategy="steps",
                                      num_train_epochs=4,
                                      fp16=True,
                                      save_steps=1000,
#                                      eval_steps=1000,
                                      logging_steps=1000,
                                      learning_rate=5e-4,
                                      weight_decay=1e-6,
                                      warmup_steps=1000,
                                      save_total_limit=2,
                                      ignore_data_skip=True,
                                     )
        
        
    trainer = Trainer(
                        model=model,
                        data_collator=data_collator,
                        args=training_args,
#                            compute_metrics=compute_metrics,
                        train_dataset=data_prepared,
#                            eval_dataset=libris_train_prepared,
                        tokenizer=processor.feature_extractor,
                        )
    
    
    trainer.train()
    
 
    
    
    
    
    
