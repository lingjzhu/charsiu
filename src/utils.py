#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import torch
import numpy as np

import os
import re
import json
import soundfile as sf
from g2p_en import G2p
from tqdm import tqdm
from praatio import textgrid
from collections import defaultdict, Counter

from transformers import Wav2Vec2CTCTokenizer,Wav2Vec2FeatureExtractor, Wav2Vec2Processor


dirname = os.path.dirname(__file__)
vocabname = os.path.join(dirname, 'vocab-ctc.json')

g2p = G2p()

tokenizer = Wav2Vec2CTCTokenizer(vocabname, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="")
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

mapping_phone2id = json.load(open(vocabname,'r'))
mapping_id2phone = {v:k for k,v in mapping_phone2id.items()}





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



def audio_preprocess(path,sr=16000):
    
    if sr == 16000:    
        features,fs = sf.read(path)
        assert fs == 16000
    else:
        features, _ = librosa.core.load(path,sr=sr)
    return processor(features, sampling_rate=16000).input_values.squeeze()



def ctc2duration(phones,resolution=0.01):
    counter = 0
    out = []
    for p,group in groupby(phones):
        length = len(list(group))
        out.append((round(counter*resolution,2),round((counter+length)*resolution,2),p))
        counter += length
        
    merged = []
    for i, (s,e,p) in enumerate(out):
        if i==0 and p=='[PAD]':
            merged.append((s,e,'[SIL]'))
        elif p=='[PAD]':
            merged.append((out[i-1][0],e,out[i-1][2]))
        elif i==len(out)-1:
            merged.append((s,e,p))
    return merged


def duration2textgrid(duration_seq,save_path=None):
    tg = textgrid.Textgrid()
    phoneTier = textgrid.IntervalTier('phones', duration_seq, 0, duration_seq[-1][1])
    tg.addTier(phoneTier)
    if save_path:
        tg.save(save_path,format="short_textgrid", includeBlankSpaces=False)
    return tg



def get_boundaries(phone_seq):
    boundaries = defaultdict(set)
    for s,e,p in phone_seq:
        boundaries[s].update([p.upper()])
#        boundaries[e].update([p.upper()+'_e'])
    timings = np.array(list(boundaries.keys()))
    symbols = list(boundaries.values())
    return (timings,symbols)


def check_textgrid_duration(textgrid,duration):
    
    endtime = textgrid.tierDict['phones'].entryList[-1].end
    if not endtime==duration:
        last = textgrid.tierDict['phones'].entryList.pop()
        textgrid.tierDict['phones'].entryList.append(last._replace(end=duration))
        
    return textgrid
    

def textgrid_to_labels(phones,duration,resolution):
    labels = []
    clock = 0.0

    for i, (s,e,p) in enumerate(phones):

        assert clock >= s
        while clock <= e:
            labels.append(p)
            clock += resolution
        
        # if more than half of the current frame is outside the current phone
        # we'll label it as the next phone
        if np.abs(clock-e) > resolution/2:
            labels[-1] = phones[min(len(phones)-1,i+1)][2]
    
    # if the final time interval is longer than the total duration
    # we will chop off this frame
    if clock-duration > resolution/2:
        labels.pop()

    return labels

def remove_null_and_numbers(labels):
    out = []
    noises = set(['SPN','NSN','LAU'])
    for l in labels:
        l = re.sub(r'\d+','',l)
        l = l.upper()
        if l == '' or l == 'SIL':
            l = '[SIL]'
        if l == 'SP':
            l = '[SIL]'
        if l in noises:
            l = '[UNK]'
        out.append(l)
    return out


def insert_sil(phones):
    
    out = []
    for i,(s,e,p) in enumerate(phones):
        
        if out:
            if out[-1][1]!=s:
                out.append((out[-1][1],s,'[SIL]'))
        out.append((s,e,p))
    return out


if __name__ == '__main__':
    pass
