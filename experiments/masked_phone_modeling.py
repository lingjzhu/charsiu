import re
from g2p_en import G2p
import json
import torch
import argparse
from dataclasses import dataclass, field
from transformers import BertTokenizerFast
from transformers import BertForMaskedLM, BertConfig
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
from transformers import Trainer,TrainingArguments
from datasets import load_dataset
from typing import Any, Dict, List, Optional, Union
from torch.utils.data import DataLoader



def get_phones(sen):
    '''
    convert texts to phone sequence
    '''
    sen = re.sub(r'\d','',sen)
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


def torch_mask_tokens(inputs, special_tokens_mask,mlm_probability=0.2):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """

    labels = inputs.clone()
    # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    # We'll use the attention mask here
    special_tokens_mask = special_tokens_mask.bool()

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    # 
    inputs[indices_replaced] = torch.tensor(mapping_phone2id['[MASK]'])

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(mapping_phone2id), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels



@dataclass
class DataCollatorMPMWithPadding:
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
        label_features = [{"input_ids": get_phone_ids(get_phones(feature["input_ids"]))[:self.max_length_labels]} for feature in features]


        with self.processor.as_target_processor():
            batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_attention_mask=self.return_attention_mask,
                return_tensors="pt",
            )
        
        inputs, labels = torch_mask_tokens(batch['input_ids'],1-batch['attention_mask'])
        # replace padding with -100 to ignore loss correctly
        labels = labels.masked_fill(batch.attention_mask.ne(1), -100)
        
        batch['input_ids'] = inputs
        batch["labels"] = labels

        return batch


g2p = G2p()
mapping_phone2id = json.load(open("./vocab.json",'r'))
mapping_id2phone = {v:k for k,v in mapping_phone2id.items()}

tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="")
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/scratch/lingjzhu_root/lingjzhu1/lingjzhu/asr')
    parser.add_argument('--out_dir',type=str,default="./models/bert-phones")

    args = parser.parse_args()


    # load text dataset
    books = load_dataset('bookcorpus',cache_dir=args.data_dir)
    books = books.rename_column('text','input_ids')

    # load model
    config = BertConfig()
    config.vocab_size = len(mapping_phone2id)
    config.hidden_size = 384
    config.num_hidden_layers = 4
    config.intermediate_size=768
    model = BertForMaskedLM(config)

    data_collator = DataCollatorMPMWithPadding(processor=processor, padding=True)

    training_args = TrainingArguments(
                                      output_dir=args.out_dir,
                                      group_by_length=True,
                                      per_device_train_batch_size=64,
                                      gradient_accumulation_steps=32,
                                      num_train_epochs=1,
                                      fp16=True,
                                      save_steps=2000,
                                      logging_steps=2000,
                                      learning_rate=1e-3,
                                      weight_decay=0.00001,
                                      warmup_steps=1000,
                                      save_total_limit=2,
                                      ignore_data_skip=True
                                     )
        
        
    trainer = Trainer(
                        model=model,
                        data_collator=data_collator,
                        args=training_args,
                        train_dataset=books['train'],
 #                       eval_dataset=books['val'], 
                     )
    
    
    trainer.train(resume_from_checkpoint='models/bert-phones/checkpoint-10000')



