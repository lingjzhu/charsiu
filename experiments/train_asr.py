#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import re
import json
import transformers
import soundfile as sf
import jiwer
import torch
import argparse

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from g2p_en import G2p
import numpy as np
from datasets import concatenate_datasets, load_dataset, load_metric, load_from_disk
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Trainer,TrainingArguments
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2ForCTC


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

def prepare_dataset(batch):
    # check that all files have the correct sampling rate

    batch["input_values"] = batch['file']
    batch["labels"] = batch['text']
    return batch

def prepare_common_voice_dataset(batch):
    # check that all files have the correct sampling rate

    batch["input_values"] = re.search(r'(.*?)\.mp3', batch['path']).group(1)+'.wav'
    batch['labels'] = batch['sentence']
    return batch

def audio_preprocess(path):
    
    features, sr = sf.read(path)
    return processor(features, sampling_rate=16000).input_values.squeeze()



@dataclass
class DataCollatorCTCWithPadding:
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
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": audio_preprocess(feature["input_values"])} for feature in features]
        label_features = [{"input_ids": get_phone_ids(get_phones(feature["labels"]))} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors="pt",
        )

        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_attention_mask=True,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch
        
wer_metric = load_metric("wer")
        
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}
    
    


def map_to_result(batch):
    model.to("cuda")
    input_values = processor(
      batch["speech"], 
      sampling_rate=16000, 
      return_tensors="pt"
        ).input_values.to("cuda")

    with torch.no_grad():
        logits = model(input_values).logits

    pred_ids = torch.argmax(logits, dim=-1)
    batch["pred_str"] = processor.batch_decode(pred_ids)[0]

    return batch

g2p = G2p()

mapping_phone2id = json.load(open("vocab-ctc.json",'r'))
mapping_id2phone = {v:k for k,v in mapping_phone2id.items()}


tokenizer = Wav2Vec2CTCTokenizer("vocab-ctc.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="")
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default='/gpfs/accounts/lingjzhu_root/lingjzhu1/lingjzhu/asr/librispeech_train360')
    parser.add_argument('--val_data', default=None,type=str)
    parser.add_argument('--test_data',default=None,type=str)
    parser.add_argument('--out_dir',type=str,default="./models/wav2vec2-base-360")

    args = parser.parse_args()
    
    # loading  data
    libris_train = load_from_disk(args.train_data)
    libris_train_prepared = libris_train.map(prepare_dataset, remove_columns=libris_train.column_names)

    common_voice = load_from_disk('/gpfs/accounts/lingjzhu_root/lingjzhu1/lingjzhu/asr/common_voice_filtered')
    common_voice = common_voice.map(prepare_common_voice_dataset, remove_columns=common_voice.column_names)

    libris_train_prepared = concatenate_datasets([libris_train_prepared, common_voice])


    if args.val_data:
        libris_val = load_from_disk('/gpfs/accounts/lingjzhu_root/lingjzhu1/lingjzhu/asr/librispeech_val')
        libris_val_prepared = libris_train.map(prepare_dataset, remove_columns=libris_val.column_names, batch_size=8, batched=True)
        
    if args.test_data:
        libris_test = load_from_disk('/gpfs/accounts/lingjzhu_root/lingjzhu1/lingjzhu/asr/librispeech_test')
        libris_test_prepared = libris_train.map(prepare_dataset, remove_columns=libris_test.column_names, batch_size=8, batched=True)

    
    
    
    # data loader
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    

    # load model
    model = Wav2Vec2ForCTC.from_pretrained( 
                                            "facebook/wav2vec2-base", 
                                            gradient_checkpointing=True, 
                                            ctc_loss_reduction="mean", 
                                            pad_token_id=processor.tokenizer.pad_token_id,
                                            vocab_size = len(processor.tokenizer)
                                            )
#    model.wav2vec2.feature_extractor.conv_layers[6].conv.stride = (1,)
#    model.config.conv_stride[-1] = 1  
    model.freeze_feature_extractor()
    
    
    # training settings
    training_args = TrainingArguments(
                                      output_dir=args.out_dir,
                                      group_by_length=True,
                                      per_device_train_batch_size=4,
                                      gradient_accumulation_steps=32,
                                      num_train_epochs=2,
                                      fp16=True,
                                      save_steps=500,
                                      logging_steps=500,
                                      learning_rate=3e-4,
                                      weight_decay=0.00005,
                                      warmup_steps=1000,
                                      save_total_limit=2,
                                     )
        
        
    trainer = Trainer(
                        model=model,
                        data_collator=data_collator,
                        args=training_args,
                        compute_metrics=compute_metrics,
                        train_dataset=libris_train_prepared,
#                        eval_dataset=libris_val_prepared,
                        tokenizer=processor.feature_extractor,
                     )
    
    
    trainer.train()
    
    '''
    results = timit["test"].map(map_to_result)
    print("Test WER: {:.3f}".format(wer_metric.compute(predictions=results["pred_str"], references=results["target_text"])))
    show_random_elements(results.remove_columns(["speech", "sampling_rate"]))
    model.to("cuda")
    input_values = processor(timit["test"][2]["speech"], sampling_rate=timit["test"][0]["sampling_rate"], return_tensors="pt").input_values.to("cuda")

    with torch.no_grad():
        logits = model(input_values).logits

    pred_ids = torch.argmax(logits, dim=-1)

    print(timit["test"][2]["target_text"])
    # convert ids to tokens
    " ".join(processor.tokenizer.convert_ids_to_tokens(pred_ids[0].tolist()))
    '''
    
    
    
    
    
    
