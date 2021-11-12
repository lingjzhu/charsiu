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
from itertools import groupby
from librosa.sequence import dtw
from transformers import Wav2Vec2CTCTokenizer,Wav2Vec2FeatureExtractor, Wav2Vec2Processor






class CharsiuPreprocessor:
    
    def __init__(self):
        
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained('charsiu/tokenizer_en_cmu')
        feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
        self.processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        self.g2p = G2p()

    def get_phones(self,sen):
        '''
        Convert texts to phone sequence

        Parameters
        ----------
        sen : str
            A str of input sentence

        Returns
        -------
        sen_clean : list
            A list of phone sequence without stress marks
        sen : list
             A list of phone sequence with stress marks

        '''     
        
        sen = self.g2p(sen)
        sen = [re.sub(r'\d','',p) for p in sen]
        return sen
    
    
    def get_phones_and_words(self,sen):
        '''
        Incomplete
        '''
        phones = self.g2p(sen)
        punctuation = set('.,!?')
        
        punc_mapping = {'.':' [SIL]', ',':' [SIL]', '!':' [SIL]', '?':' [SIL]'}
        sen = re.sub(r'.,!?', ' [SIL]', sen)
        sen = sen.split(' ')
        return sen
    
    
    def get_phone_ids(self,phones,append_silence=True):
        '''
        Convert phone sequence to ids

        Parameters
        ----------
        phones : list
            A list of phone sequence
        append_silence : bool, optional
            Whether silence is appended at the beginning and the end of the sequence. 
            The default is True.

        Returns
        -------
        list
            A list of one-hot representations of phones

        '''
        ids = []
        punctuation = set('.,!?')
        for p in phones:
            if re.match(r'^\w+?$',p):
                ids.append(self.mapping_phone2id(p))
            elif p in punctuation:
                ids.append(self.mapping_phone2id('[SIL]'))
        # append silence at the beginning and the end
        if append_silence:
            if ids[0]!=0:
                ids = [0]+ids
            if ids[-1]!=0:
                ids.append(0)
        return ids 
    
    def mapping_phone2id(self,phone):
        '''
        Convert a phone to a numerical id

        Parameters
        ----------
        phone : str
            A phonetic symbol

        Returns
        -------
        int
            A one-hot id for the input phone

        '''
        return self.processor.tokenizer.convert_tokens_to_ids(phone)
    
    def mapping_id2phone(self,idx):
        '''
        Convert a numerical id to a phone

        Parameters
        ----------
        idx : int
            A one-hot id for a phone

        Returns
        -------
        str
            A phonetic symbol

        '''
        return self.processor.tokenizer.convert_ids_to_tokens(idx)
        
    
    def audio_preprocess(self,path,sr=16000):
        '''
        Load and normalize audio
        If the sampling rate is incompatible with models, the input audio will be resampled.

        Parameters
        ----------
        path : str
            The path to the audio
        sr : int, optional
            Audio sampling rate. The default is 16000.

        Returns
        -------
        torch.Tensor [(n,)]
            A list of audio sample as an one dimensional torch tensor

        '''
        
        if sr == 16000:    
            features,fs = sf.read(path)
            assert fs == 16000
        else:
            features, _ = librosa.core.load(path,sr=sr)
        return self.processor(features, sampling_rate=16000,return_tensors='pt').input_values.squeeze()



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


def seq2duration(phones,resolution=0.01):
    counter = 0
    out = []
    for p,group in groupby(phones):
        length = len(list(group))
        out.append((round(counter*resolution,2),round((counter+length)*resolution,2),p))
        counter += length
    return out


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


def forced_align(cost, phone_ids):
    D,align = dtw(C=-cost[:,phone_ids])

    align_seq = [-1 for i in range(max(align[:,0])+1)]
    for i in list(align):
    #    print(align)
        if align_seq[i[0]]<i[1]:
            align_seq[i[0]]=i[1]

    align_id = list(align_seq)
    return align_id



if __name__ == '__main__':
    pass
