#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import sys
import torch
from itertools import groupby
sys.path.append('src/')
import numpy as np
#sys.path.insert(0,'src')
from models import Wav2Vec2ForAttentionAlignment, Wav2Vec2ForFrameClassification, Wav2Vec2ForCTC
from utils import seq2duration,forced_align,duration2textgrid,word2textgrid
from processors import CharsiuPreprocessor_zh, CharsiuPreprocessor_en

processors = {'zh':CharsiuPreprocessor_zh,
              'en':CharsiuPreprocessor_en}

class charsiu_aligner:
    
    def __init__(self, 
                 lang='en', 
                 sampling_rate=16000, 
                 device=None,
                 recognizer=None,
                 processor=None, 
                 resolution=0.01):
                
        self.lang = lang 
        
        if processor is not None:
            self.processor = processor
        else:
            self.charsiu_processor = processors[self.lang]()
        
        
        
        self.resolution = resolution
        
        self.sr = sampling_rate
        
        self.recognizer = recognizer
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

    
    def _freeze_model(self):
        self.aligner.eval().to(self.device)
        if self.recognizer is not None:
            self.recognizer.eval().to(self.device)
    
    
    
    def align(self,audio,text):
        raise NotImplementedError()
        
        
        
    def serve(self,audio,save_to,output_format='variable',text=None):
        raise NotImplementedError()
        
    
    def _to_textgrid(self,phones,save_to):
        '''
        Convert output tuples to a textgrid file

        Parameters
        ----------
        phones : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        duration2textgrid(phones,save_path=save_to)
        print('Alignment output has been saved to %s'%(save_to))
    
    
    
    def _to_tsv(self,phones,save_to):
        '''
        Convert output tuples to a tab-separated file

        Parameters
        ----------
        phones : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        with open(save_to,'w') as f:
            for start,end,phone in phones:
                f.write('%s\t%s\t%s\n'%(start,end,phone))
        print('Alignment output has been saved to %s'%(save_to))





class charsiu_forced_aligner(charsiu_aligner):
    
    def __init__(self, aligner, sil_threshold=10, **kwargs):
        super(charsiu_forced_aligner, self).__init__(**kwargs)
        self.aligner = Wav2Vec2ForFrameClassification.from_pretrained(aligner)
        self.sil_threshold = sil_threshold
        
        self._freeze_model()
        
        
    def align(self, audio, text):
        '''
        Perform forced alignment

        Parameters
        ----------
        audio : np.ndarray [shape=(n,)]
            time series of speech signal
        text : str
            The transcription

        Returns
        -------
        A tuple of aligned phones in the form (start_time, end_time, phone)

        '''
        audio = self.charsiu_processor.audio_preprocess(audio,sr=self.sr)
        audio = torch.Tensor(audio).unsqueeze(0).to(self.device)
        phones, words = self.charsiu_processor.get_phones_and_words(text)
        phone_ids = self.charsiu_processor.get_phone_ids(phones)

        with torch.no_grad():
            out = self.aligner(audio)
        cost = torch.softmax(out.logits,dim=-1).detach().cpu().numpy().squeeze()
          

        # single out silent intervals
        preds = np.argmax(cost,axis=-1)
        sil_mask = []
        for key, group in groupby(preds):
            group = list(group)
            if  (key==1 and len(group)<=self.sil_threshold):
                sil_mask += [0 for i in range(len(group))]
            else:
                sil_mask += group
        sil_mask = np.array(sil_mask)
        nonsil_idx = np.argwhere(sil_mask!=1).squeeze()
        
        aligned_phone_ids = forced_align(cost[nonsil_idx,:],phone_ids[1:-1])
        
        aligned_phones = [self.charsiu_processor.mapping_id2phone(phone_ids[1:-1][i]) for i in aligned_phone_ids]
            # merge silent and non-silent intervals
        pred_phones = []
        count = 0
        for i in sil_mask:
            if i==1:
                pred_phones.append('[SIL]')
            else:
                pred_phones.append(aligned_phones[count])
                count += 1
        assert len(pred_phones) == len(preds)
        
        
        pred_phones = seq2duration(pred_phones,resolution=self.resolution)
        
        pred_words = self.charsiu_processor.align_words(pred_phones,phones,words)
        return pred_phones, pred_words
    
    
    def serve(self,audio,text,save_to,output_format='textgrid'):
        '''
         A wrapper function for quick inference
    
        Parameters
        ----------
        audio : TYPE
            DESCRIPTION.
        text : TYPE, optional
            DESCRIPTION. The default is None.
        output_format : str, optional
            Output phone-taudio alignment as a "tsv" or "textgrid" file. 
            The default is 'textgrid'.
    
        Returns
        -------
        None.
    
        '''
        phones, words = self.align(audio,text)

        if output_format == 'tsv':
            if save_to.endswith('.tsv'):
                save_to_phone = save_to.replace('.tsv','_phone.tsv')
                save_to_word = save_to.replace('.tsv','_word.tsv')
            else:
                save_to_phone = save_to + '_phone.tsv'
                save_to_word = save_to + '_word.tsv'
            
            self._to_tsv(phones, save_to_phone)
            self._to_tsv(words, save_to_word)
            
        elif output_format == 'textgrid':
            self._to_textgrid(phones, words, save_to)
        else:
            raise Exception('Please specify the correct output format (tsv or textgird)!')    

    def _to_textgrid(self,phones,words,save_to):
        '''
        Convert output tuples to a textgrid file

        Parameters
        ----------
        phones : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        word2textgrid(phones,words,save_path=save_to)
        print('Alignment output has been saved to %s'%(save_to))
    
    



class charsiu_attention_aligner(charsiu_aligner):
    

    def __init__(self, aligner, **kwargs):
        super(charsiu_attention_aligner, self).__init__(**kwargs)
        self.aligner = Wav2Vec2ForAttentionAlignment.from_pretrained(aligner)
    
        self._freeze_model()
        
        
    def align(self, audio, text):
        '''
        Perform forced alignment

        Parameters
        ----------
        audio : np.ndarray [shape=(n,)]
            time series of speech signal
        text : str
            The transcription

        Returns
        -------
        A tuple of aligned phones in the form (start_time, end_time, phone)

        '''
        audio = self.charsiu_processor.audio_preprocess(audio,sr=self.sr)
        audio = torch.Tensor(audio).unsqueeze(0).to(self.device)
        phones, words = self.charsiu_processor.get_phones_and_words(text)
        phone_ids = self.charsiu_processor.get_phone_ids(phones)

        
        batch = {'input_values':audio,
                 'labels': torch.tensor(phone_ids).unsqueeze(0).long().to(self.device)
                }
        
        with torch.no_grad():
          out = self.aligner(**batch)
              
        att = torch.softmax(out.logits,dim=-1),
        preds = torch.argmax(att[0],dim=-1).cpu().detach().squeeze().numpy()
        pred_phones = [self.charsiu_processor.mapping_id2phone(phone_ids[i]) for i in preds]
        pred_phones = seq2duration(pred_phones,resolution=self.resolution)
            
        return pred_phones    
    
    def serve(self,audio,text,save_to,output_format='textgrid'):
        '''
         A wrapper function for quick inference
    
        Parameters
        ----------
        audio : TYPE
            DESCRIPTION.
        text : TYPE, optional
            DESCRIPTION. The default is None.
        output_format : str, optional
            Output phone-taudio alignment as a "tsv" or "textgrid" file. 
            The default is 'textgrid'.
    
        Returns
        -------
        None.
    
        '''
        aligned_phones = self.align(audio,text)

        if output_format == 'tsv':
            self._to_tsv(aligned_phones, save_to)
        elif output_format == 'textgrid':
            self._to_textgrid(aligned_phones, save_to)
        else:
            raise Exception('Please specify the correct output format (tsv or textgird)!')
    
    
    def _to_textgrid(self,phones,save_to):
        '''
        Convert output tuples to a textgrid file

        Parameters
        ----------
        phones : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        duration2textgrid(phones,save_path=save_to)
        print('Alignment output has been saved to %s'%(save_to))
    


    
    
    
class charsiu_chain_attention_aligner(charsiu_aligner):
    
    def __init__(self, aligner, recognizer, **kwargs):
        super(charsiu_chain_attention_aligner, self).__init__(**kwargs)
        self.aligner = Wav2Vec2ForAttentionAlignment.from_pretrained(aligner)
        self.recognizer = Wav2Vec2ForCTC.from_pretrained(recognizer)
        
        self._freeze_model()
        
    def align(self, audio):
        '''
        Recognize phones and perform forced alignment

        Parameters
        ----------
        audio : np.ndarray [shape=(n,)]
            time series of speech signal

        Returns
        -------
        A tuple of aligned phones in the form (start_time, end_time, phone)

        '''
        if self.recognizer is None:
            print('A recognizer is not specified. Will use the default recognizer.')
            self.recognizer = Wav2Vec2ForCTC.from_pretrained('charsiu/en_w2v2_ctc_libris_and_cv')
        
        # perform phone recognition
        audio = self.charsiu_processor.audio_preprocess(audio,sr=self.sr)
        audio = torch.tensor(audio).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            out = self.recognizer(audio)
            
        pred_ids = torch.argmax(out.logits,dim=-1).squeeze()
        phones = self.charsiu_processor.processor.tokenizer.convert_ids_to_tokens(pred_ids,skip_special_tokens=True)
        phones = [p for p,group in groupby(phones)]
        phone_ids = self.charsiu_processor.get_phone_ids(phones)
        
        # perform forced alignment
        batch = {'input_values':audio,
         'labels': torch.tensor(phone_ids).unsqueeze(0).long().to(self.device)
        }

        with torch.no_grad():
          out = self.aligner(**batch)
        att = torch.softmax(out.logits,dim=-1)
        
        preds = torch.argmax(att[0],dim=-1).cpu().detach().squeeze().numpy()
        pred_phones = [self.charsiu_processor.mapping_id2phone(phone_ids[i]) for i in preds]
        pred_phones = seq2duration(pred_phones,resolution=self.resolution)
        return pred_phones
    
    
    def serve(self,audio,save_to,output_format='textgrid'):
        '''
         A wrapper function for quick inference
         Note. Only phones are supported in text independent alignment.
         
        Parameters
        ----------
        audio : TYPE
            DESCRIPTION.
        text : TYPE, optional
            DESCRIPTION. The default is None.
        output_format : str, optional
            Output phone-taudio alignment as a "tsv" or "textgrid" file. 
            The default is 'textgrid'.
    
        Returns
        -------
        None.
    
        '''
        aligned_phones = self.align(audio)

        if output_format == 'tsv':
            self._to_tsv(aligned_phones, save_to)
        elif output_format == 'textgrid':
            self._to_textgrid(aligned_phones, save_to)
        else:
            raise Exception('Please specify the correct output format (tsv or textgird)!')



class charsiu_chain_forced_aligner(charsiu_aligner):
    
    def __init__(self, aligner, recognizer, **kwargs):
        super(charsiu_chain_forced_aligner, self).__init__(**kwargs)
        self.aligner = Wav2Vec2ForFrameClassification.from_pretrained(aligner)
        self.recognizer = Wav2Vec2ForCTC.from_pretrained(recognizer)
        
        self._freeze_model()
        
    def align(self, audio):
        '''
        Recognize phones and perform forced alignment

        Parameters
        ----------
        audio : np.ndarray [shape=(n,)]
            time series of speech signal

        Returns
        -------
        A tuple of aligned phones in the form (start_time, end_time, phone)

        '''
        if self.recognizer is None:
            print('A recognizer is not specified. Will use the default recognizer.')
            self.recognizer = Wav2Vec2ForCTC.from_pretrained('charsiu/en_w2v2_ctc_libris_and_cv')
        
        # perform phone recognition
        audio = self.charsiu_processor.audio_preprocess(audio,sr=self.sr)
        audio = torch.tensor(audio).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.recognizer(audio)
            
        pred_ids = torch.argmax(out.logits,dim=-1).squeeze()
        phones = self.charsiu_processor.processor.tokenizer.convert_ids_to_tokens(pred_ids,skip_special_tokens=True)
        phones = [p for p,group in groupby(phones)]
        phone_ids = self.charsiu_processor.get_phone_ids(phones)
        
        # perform forced alignment
        with torch.no_grad():
            out = self.aligner(audio)
        cost = torch.softmax(out.logits,dim=-1).detach().cpu().numpy().squeeze()
          
        aligned_phone_ids = forced_align(cost,phone_ids)
        
        aligned_phones = [self.charsiu_processor.mapping_id2phone(phone_ids[i]) for i in aligned_phone_ids]
        pred_phones = seq2duration(aligned_phones,resolution=self.resolution)
        return pred_phones
    
    
    def serve(self,audio,save_to,output_format='textgrid'):
        '''
         A wrapper function for quick inference
         Note. Only phones are supported in text independent alignment.
         
        Parameters
        ----------
        audio : TYPE
            DESCRIPTION.
        text : TYPE, optional
            DESCRIPTION. The default is None.
        output_format : str, optional
            Output phone-taudio alignment as a "tsv" or "textgrid" file. 
            The default is 'textgrid'.
    
        Returns
        -------
        None.
    
        '''
        aligned_phones = self.align(audio)

        if output_format == 'tsv':
            self._to_tsv(aligned_phones, save_to)
        elif output_format == 'textgrid':
            self._to_textgrid(aligned_phones, save_to)
        else:
            raise Exception('Please specify the correct output format (tsv or textgird)!')



class charsiu_predictive_aligner(charsiu_aligner):
    
    def __init__(self, aligner, **kwargs):
        super(charsiu_predictive_aligner, self).__init__(**kwargs)
        self.aligner = Wav2Vec2ForFrameClassification.from_pretrained(aligner)
        self._freeze_model()
    
    def align(self, audio):
        '''
        Directly predict the phone-to-audio alignment based on acoustic signal only

        Parameters
        ----------
        audio : np.ndarray [shape=(n,)]
            time series of speech signal

        Returns
        -------
        A tuple of aligned phones in the form (start_time, end_time, phone)

        '''
        audio = self.charsiu_processor.audio_preprocess(audio,sr=self.sr)
        audio = torch.Tensor(audio).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.aligner(audio)
            
        pred_ids = torch.argmax(out.logits.squeeze(),dim=-1)
        pred_ids = pred_ids.detach().cpu().numpy()
        pred_phones = [self.charsiu_processor.mapping_id2phone(int(i)) for i in pred_ids]
        pred_phones = seq2duration(pred_phones,resolution=self.resolution)
        return pred_phones
    

    def serve(self,audio,save_to,output_format='textgrid'):
        '''
         A wrapper function for quick inference
         Note. Only phones are supported in text independent alignment.
         
        Parameters
        ----------
        audio : TYPE
            DESCRIPTION.
        text : TYPE, optional
            DESCRIPTION. The default is None.
        output_format : str, optional
            Output phone-taudio alignment as a "tsv" or "textgrid" file. 
            The default is 'textgrid'.
    
        Returns
        -------
        None.
    
        '''
        aligned_phones = self.align(audio)

        if output_format == 'tsv':
            self._to_tsv(aligned_phones, save_to)
        elif output_format == 'textgrid':
            self._to_textgrid(aligned_phones, save_to)
        else:
            raise Exception('Please specify the correct output format (tsv or textgird)!')


if __name__ == "__main__":
    
    '''
    Test code
    '''
    charsiu = charsiu_forced_aligner(aligner='charsiu/en_w2v2_fc_10ms')
    # perform forced alignment
    alignment = charsiu.align(audio='./local/SA1.WAV',
                              text='She had your dark suit in greasy wash water all year.')

    # Chinese models
    charsiu = charsiu_predictive_aligner(aligner='charsiu/zh_xlsr_fc_10ms',lang='zh')
    charsiu.align(audio=audio)
    charsiu.serve(audio='./local/SSB00050015_16k.wav', save_to='./local/SSB00050015.TextGrid')
    
    charsiu = charsiu_forced_aligner(aligner='charsiu/zh_w2v2_tiny_fc_10ms',lang='zh')
    phones, words = charsiu.align(audio='/home/lukeum/Downloads/000001_16k.wav',text='卡尔普陪外孙玩滑梯。')
    charsiu.serve(audio='./local/SSB00050015_16k.wav', text='经广州日报报道后成为了社会热点。',
                  save_to='./local/SSB00050015.TextGrid')
    

    audio = '/mnt/exp/data_aishell/wav/train/S0104/BAC009S0104W0126.wav'
    text = '二线 城市 购入 土地 二千 九百五十九 万平 米'

    charsiu.serve(audio=audio, text=text,
              save_to='./local/sample.TextGrid')