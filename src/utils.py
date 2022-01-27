#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np

import re
from praatio import textgrid
from itertools import groupby
from librosa.sequence import dtw



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


def word2textgrid(duration_seq,word_seq,save_path=None):
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
    wordTier = textgrid.IntervalTier('words', word_seq, 0, word_seq[-1][1])
    tg.addTier(wordTier)
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
    
    D,align = dtw(C=-cost[:,phone_ids],
                  step_sizes_sigma=np.array([[1, 1], [1, 0]]))

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

    pass 











