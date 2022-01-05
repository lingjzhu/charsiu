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
from g2pM import G2pM
from tqdm import tqdm
from praatio import textgrid
from collections import defaultdict, Counter
from itertools import groupby, chain
from librosa.sequence import dtw
from transformers import Wav2Vec2CTCTokenizer,Wav2Vec2FeatureExtractor, Wav2Vec2Processor






class CharsiuPreprocessor_en:
    
    def __init__(self):
        
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained('charsiu/tokenizer_en_cmu')
        feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
        self.processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        self.g2p = G2p()
        self.sil = self.mapping_phone2id('[SIL]')
        
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
             
    
        xxxxx should sen_clean be deleted?

        '''     
        
        sen = self.g2p(sen)
        sen = [re.sub(r'\d','',p) for p in sen]
        return sen
    
    
    def get_phones_and_words(self,sen):
        '''
        Convert texts to words then to phone sequence

        Parameters
        ----------
        sen : str
            A str of input sentence

        Returns
        -------
        sen : list
             A list of phone sequence with stress marks
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
        ids: list
            A list of one-hot representations of phones

        '''
        
        ids = []
        punctuation = set('.,!?。，！？')
        for p in phones:
            if re.match(r'^\w+?$',p):
                ids.append(self.mapping_phone2id(p))
            elif p in punctuation:
                ids.append(self.mapping_phone2id('[SIL]'))
        # append silence at the beginning and the end
        if append_silence:
            if ids[0]!=self.sil:
                ids = [self.sil]+ids
            if ids[-1]!=self.sil:
                ids.append(self.sil)
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


'''
Object for Mandarin g2p processor
'''


consonant_list = set(['b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k',
                  'h', 'j', 'q', 'x', 'zh', 'ch', 'sh', 'r', 'z',
                  'c', 's'])

transform_dict = {'ju':'jv', 'qu':'qv', 'xu':'xv','jue':'jve',
                  'que':'qve', 'xue':'xve','quan':'qvan',
                  'xuan':'xvan','juan':'jvan',
                  'qun':'qvn','xun':'xvn', 'jun':'jvn',
                     'yuan':'van', 'yue':'ve', 'yun':'vn',
                    'you':'iou', 'yan':'ian', 'yin':'in',
                    'wa':'ua', 'wo':'uo', 'wai':'uai',
                    'weng':'ueng', 'wang':'uang','wu':'u',
                    'yu':'v','yi':'i','yo':'io','ya':'ia', 'ye':'ie', 
                    'yao':'iao','yang':'iang', 'ying':'ing', 'yong':'iong',
                    'yvan':'van', 'yve':'ve', 'yvn':'vn',
                    'wa':'ua', 'wo':'uo', 'wai':'uai',
                    'wei':'ui', 'wan':'uan', 'wen':'un', 
                    'weng':'ueng', 'wang':'uang','yv':'v',
                    'wuen':'un','wuo':'uo','wuang':'uang',
                    'wuan':'uan','wua':'ua','wuai':'uai',
                    'zhi':'zhiii','chi':'chiii','shi':'shiii',
                    'zi':'zii','ci':'cii','si':'sii'}


class CharsiuPreprocessor_zh(CharsiuPreprocessor_en):

    def __init__(self):
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained('charsiu/tokenizer_zh_pinyin')
        feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
        self.processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        self.g2p = G2pM()
        self.sil = self.mapping_phone2id('[SIL]')
        
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
        
        xxxxx should sen_clean be removed?
        '''     
        
        sen = self.g2p(sen)
        sen = [self._separate_syllable(transform_dict.get(p[:-1],p[:-1])+p[-1]) if re.search(r'\w+\d',p) else p for p in sen ]
        return list(chain.from_iterable(sen))

    def _separate_syllable(self,syllable):
        """
        seprate syllable to consonant + ' ' + vowel

        Parameters
        ----------
        syllable : xxxxx TYPE
            xxxxx DESCRIPTION.

        Returns
        -------
        syllable: xxxxx TYPE
            xxxxxx DESCRIPTION.

        """
        
        assert syllable[-1].isdigit()
        if syllable == 'ri4':
            return ('r','iii4')
        if syllable in er_mapping.keys():
            return er_mapping[syllable]
        if syllable[0:2] in consonant_list:
            #return syllable[0:2].encode('utf-8'),syllable[2:].encode('utf-8')
            return syllable[0:2], syllable[2:]
        elif syllable[0] in consonant_list:
            #return syllable[0].encode('utf-8'),syllable[1:].encode('utf-8')
            return syllable[0], syllable[1:]
        else:
            #return (syllable.encode('utf-8'),)
            return (syllable,)
        

er_mapping ={'er1':('e1','rr'),'er2':('e2','rr'),'er3':('e3','rr'),'er4':('e4','rr'),'er5':('e5','rr')}


def ctc2duration(phones,resolution=0.01):
    """
    xxxxx convert ctc to duration

    Parameters
    ----------
    phones : list
        A list of phone sequence
    resolution : float, optional
        The resolution of xxxxx. The default is 0.01.

    Returns
    -------
    merged : list
        xxxxx A list of duration values.

    """
    
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
    """
    xxxxx convert phone sequence to duration

    Parameters
    ----------
    phones : list
        A list of phone sequence
    resolution : float, optional
        The resolution of xxxxx. The default is 0.01.

    Returns
    -------
    out : list
        xxxxx A list of duration values.

    """
    
    counter = 0
    out = []
    for p,group in groupby(phones):
        length = len(list(group))
        out.append((round(counter*resolution,2),round((counter+length)*resolution,2),p))
        counter += length
    return out


def duration2textgrid(duration_seq,save_path=None):
    """
    Save duration values to textgrids

    Parameters
    ----------
    duration_seq : list
        xxxxx A list of duration values.
    save_path : str, optional
        The path to save the TextGrid files. The default is None.

    Returns
    -------
    tg : TextGrid file?? str?? xxxxx?
        A textgrid object containing duration information.

    """

    tg = textgrid.Textgrid()
    phoneTier = textgrid.IntervalTier('phones', duration_seq, 0, duration_seq[-1][1])
    tg.addTier(phoneTier)
    if save_path:
        tg.save(save_path,format="short_textgrid", includeBlankSpaces=False)
    return tg



def get_boundaries(phone_seq):
    """
    Get time of phone boundaries

    Parameters
    ----------
    phone_seq : list xxxx?
        A list of phone sequence.

    Returns
    -------
    timings: A list of time stamps
    symbols: A list of phone symbols

    """
    
    boundaries = defaultdict(set)
    for s,e,p in phone_seq:
        boundaries[s].update([p.upper()])
#        boundaries[e].update([p.upper()+'_e'])
    timings = np.array(list(boundaries.keys()))
    symbols = list(boundaries.values())
    return (timings,symbols)


def check_textgrid_duration(textgrid,duration):
    """
    Check whether the duration of a textgrid file equals to 'duration'. 
    If not, replace duration of the textgrid file.

    Parameters
    ----------
    textgrid : .TextGrid object
        A .TextGrid object.
    duration : float
        A given length of time.

    Returns
    -------
    textgrid : .TextGrid object
        A modified/unmodified textgrid.

    """
    
    
    endtime = textgrid.tierDict['phones'].entryList[-1].end
    if not endtime==duration:
        last = textgrid.tierDict['phones'].entryList.pop()
        textgrid.tierDict['phones'].entryList.append(last._replace(end=duration))
        
    return textgrid
    

def textgrid_to_labels(phones,duration,resolution):
    """
    

    Parameters
    ----------
    phones : list
        A list of phone sequence
    resolution : float, optional
        The resolution of xxxxx. The default is 0.01.
    duration : float
        A given length of time.
    

    Returns
    -------
    labels : list
        A list of phone labels.

    """
    
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
    """
    Remove labels which are null, noise, or numbers.

    Parameters
    ----------
    labels : list
        A list of text labels.

    Returns
    -------
    out : list
        A list of new labels.

    """
    
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
    """
    Insert silences.

    Parameters
    ----------
    phones : list
        A list of phone sequence

    Returns
    -------
    out : list
        A list of new labels.

    """
    
    out = []
    for i,(s,e,p) in enumerate(phones):
        
        if out:
            if out[-1][1]!=s:
                out.append((out[-1][1],s,'[SIL]'))
        out.append((s,e,p))
    return out


def forced_align(cost, phone_ids):
    """
    Force align text to audio.

    Parameters
    ----------
    cost : float xxxxx
        xxxxx.
    phone_ids : list
        A list of phone IDs.

    Returns
    -------
    align_id : list
        A list of IDs for aligned phones.

    """
    
    D,align = dtw(C=-cost[:,phone_ids])

    align_seq = [-1 for i in range(max(align[:,0])+1)]
    for i in list(align):
    #    print(align)
        if align_seq[i[0]]<i[1]:
            align_seq[i[0]]=i[1]

    align_id = list(align_seq)
    return align_id



if __name__ == '__main__':
    '''
    Testing functions
    '''    

    processor = CharsiuPreprocessor_zh()
    phones = processor.get_phones("鱼香肉丝、王道椒香鸡腿和川蜀鸡翅。")    
    print(phones)
    ids = processor.get_phone_ids(phones)
    print(ids)

    phones = processor.get_phones("聚集 了 东郊 某 中学 的 学生 二十多 人。")    
    print(phones)
    ids = processor.get_phone_ids(phones)
    print(ids)

    processor = CharsiuPreprocessor_en()
    phones = processor.get_phones("charsiu phonetic aligner")    
    print(phones)
    ids = processor.get_phone_ids(phones)
    print(ids)











