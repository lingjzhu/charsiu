#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import re
import transformers
import soundfile as sf
import torch
import json
import numpy as np

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from g2p_en import G2p
from datasets import load_dataset, load_metric, load_from_disk
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Trainer,TrainingArguments
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2ForCTC, Wav2Vec2Config
from transformers.modeling_outputs import CausalLMOutput, MaskedLMOutput


def prepare_dataset_20ms(batch):

    batch["input_values"] = batch['file']
    batch["labels"] = [mapping_phone2id[p] for p in batch['frame_labels']]
    assert len(batch['frame_labels']) == len(batch['labels'])
    return batch


def prepare_dataset_10ms(batch):

    batch["input_values"] = batch['file']
#    batch["labels"] = [mapping_phone2id[p] for p in batch['frame_labels_10ms']]
    batch["labels"] = [mapping_phone2id[p] for p in batch['labels']]
    assert len(batch['frame_labels_10ms']) == len(batch['labels'])
    return batch
	
def prepare_test_dataset_10ms(batch):

    batch["input_values"] = batch['file']
    batch["labels"] = [mapping_phone2id[p] for p in batch['frame_labels_10ms']]
#    batch["labels"] = [mapping_phone2id[p] for p in batch['labels']]
    assert len(batch['frame_labels_10ms']) == len(batch['labels'])
    return batch

def prepare_dataset_cv(batch):

    batch["input_values"] = batch['path'].replace('.mp3','.wav')
    batch["labels"] = [mapping_phone2id[p] for p in batch['labels']]
    assert len(batch['labels']) == len(batch['labels'])
    return batch


def audio_preprocess(path):
    
    features,sr = sf.read(path)
    assert sr == 16000
    return processor(features, sampling_rate=16000).input_values.squeeze()



@dataclass
class DataCollatorClassificationWithPadding:
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
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_attention_mask=self.return_attention_mask,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

class Wav2Vec2ForFrameClassification(Wav2Vec2ForCTC):
    
    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:

            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            loss = torch.nn.functional.cross_entropy(logits.view(-1,logits.size(2)), labels.flatten(), reduction="mean")

            

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

#    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    comparison = (pred_ids == pred.label_ids)
    comparison = comparison[pred.label_ids != -100].flatten()
    acc = np.sum(comparison)/len(comparison)

    return {"phone_accuracy": acc}
    
    




tokenizer = Wav2Vec2CTCTokenizer("./dict/vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="")
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

mapping_phone2id = json.load(open("./dict/vocab.json",'r'))
mapping_id2phone = {v:k for k,v in mapping_phone2id.items()}



if __name__ == "__main__":

    frameshift = 10
    
    if frameshift == 10:
        prepare_dataset = prepare_dataset_10ms
#        prepare_dataset = prepare_dataset_cv
    else:
        prepare_dataset = prepare_dataset_20ms
    
    # loading  data
#    libris = load_from_disk('/shared/2/datasets/speech/librispeech_asr/librispeech_full')
#    libris = load_from_disk('/shared/2/datasets/speech/common_voice/common_voice_align')
    libris = load_from_disk('/shared/2/datasets/speech/librispeech_asr/librispeech_360_align')
    libris_train_prepared = libris.map(prepare_dataset,batched=False)
    

#    libris_train_prepared = libris_prepared.filter(lambda x: bool(re.search('train-clean-360',x['file'])))

    libris_val = load_from_disk('/shared/2/datasets/speech/librispeech_asr/librispeech_full_test')
    libris_val = libris_val.select([i for i in range(200)])
    libris_val_prepared = libris_val.map(prepare_test_dataset_10ms,batched=False)
    
 
#    libris_test_prepared = libris_prepared.filter(lambda x: bool(re.search('test-clean',x['file'])))

    
    
    
    # data loader
    data_collator = DataCollatorClassificationWithPadding(processor=processor, padding=True)
    
    mode = 'base'
	
    # load model 
    if mode == 'tiny':    
        config = Wav2Vec2Config()
        config.num_attention_heads = 6
        config.hidden_size = 384
        config.num_hidden_layers = 6
        config.vocab_size = len(processor.tokenizer)
        model = Wav2Vec2ForFrameClassification(config)  

    # load pretrained weights
    
        pretrained_model = Wav2Vec2ForFrameClassification.from_pretrained( 
                                                "facebook/wav2vec2-base", 
                                                gradient_checkpointing=True, 
                                                pad_token_id=processor.tokenizer.pad_token_id,
                                                vocab_size = len(processor.tokenizer)
                                                )

        layers = {'wav2vec2.feature_extractor.conv_layers.0.conv.weight', 'wav2vec2.feature_extractor.conv_layers.0.layer_norm.weight', 'wav2vec2.feature_extractor.conv_layers.0.layer_norm.bias', 'wav2vec2.feature_extractor.conv_layers.1.conv.weight', 'wav2vec2.feature_extractor.conv_layers.2.conv.weight', 'wav2vec2.feature_extractor.conv_layers.3.conv.weight', 'wav2vec2.feature_extractor.conv_layers.4.conv.weight', 'wav2vec2.feature_extractor.conv_layers.5.conv.weight', 'wav2vec2.feature_extractor.conv_layers.6.conv.weight','quantizer.codevectors', 'quantizer.weight_proj.weight', 'quantizer.weight_proj.bias', 'project_q.weight', 'project_q.bias'}
        pretrained_dict = {k: v for k, v in  pretrained_model.state_dict().items() if k in layers}
        del pretrained_model
        
        # update pretrained weights
        state_dict = model.state_dict()
        state_dict.update(pretrained_dict)
        model.load_state_dict(state_dict)
        
    elif mode == 'base':
        model = Wav2Vec2ForFrameClassification.from_pretrained( 
                                                "facebook/wav2vec2-base", 
                                                gradient_checkpointing=True, 
                                                pad_token_id=processor.tokenizer.pad_token_id,
                                                vocab_size = len(processor.tokenizer)
                                                )
    

    # freeze convolutional layers and set the stride of the last conv layer to 1
    # this increase the sampling frequency to 98 Hz
    model.wav2vec2.feature_extractor.conv_layers[6].conv.stride = (1,)
    model.config.conv_stride[-1] = 1    
    model.freeze_feature_extractor()

    
    # training settings
    training_args = TrainingArguments(
                                      output_dir="/shared/2/projects/phone_segmentation/models/wav2vec2-base-FC-10ms-libris-iter3",
                                      group_by_length=True,
                                      per_device_train_batch_size=2,
                                      per_device_eval_batch_size=4,
                                      gradient_accumulation_steps=8,
                                      evaluation_strategy="steps",
                                      num_train_epochs=2,
                                      fp16=True,
                                      save_steps=500,
                                      eval_steps=500,
                                      logging_steps=500,
                                      learning_rate=3e-4,
                                      weight_decay=0.0001,
                                      warmup_steps=1000,
                                      save_total_limit=2,
                                     )
        
        
    trainer = Trainer(
                        model=model,
                        data_collator=data_collator,
                        args=training_args,
                        compute_metrics=compute_metrics,
                        train_dataset=libris_train_prepared,
                        eval_dataset=libris_val_prepared,
                        tokenizer=processor.feature_extractor,
                     )
    
    
    trainer.train()
    
  
