# Forced-aligned Speech Datasets

Here we release **TextGrids** for several datasets that have been forced aligned with [Charsiu Forced Aligner](https://github.com/lingjzhu/charsiu). Hoepfully they might be helpful for your research. 
Forced alignment does not generate perfect alignments. **Use at you own discrection**.  
[English](data.md#alignments-for-english-datasets)  
[Mandarin](data.md#alignments-for-mandarin-speech-datasets)  


Please cite this if you use these alignments in your research projects.
```
@article{zhu2022charsiu,
  title={Phone-to-audio alignment without text: A Semi-supervised Approach},
  author={Zhu, Jian and Zhang, Cong and Jurgens, David},
  journal={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2022},
  url=https://arxiv.org/abs/2110.03876,
 }
```

## Alignments for English datasets

### Textgrids
You can find [all textgrids for the trianing sets (~860k utterances) in this google drive folder](https://drive.google.com/drive/folders/1IF0WB5-8VXfaENtE4r5rehHHK8YFe61S?usp=sharing). It contains phone- and word-level alignments for the English subset of Common Voice (~2000 hours). It was aligned with `charsiu_forced_aligner` with the model `charsiu/en_w2v2_fc_10ms`. 
All filenames are matched. Only a few mismatched samples were discarded. 

### Audio
You can find the dataset at the [Common Voice Project](https://commonvoice.mozilla.org/en/datasets).  

The audio data can also be easily accessed through [the Common Voce page](https://huggingface.co/datasets/mozilla-foundation/common_voice_8_0) at HuggingFace hub. An account is needed for authentication.
```
from datasets import load_dataset

dataset = load_dataset("mozilla-foundation/common_voice_8_0", "en", split='train',use_auth_token=True)
```
Note that ~80GB of memory is needed to load the dataset into memory.

Please cite Common Voice if you use this dataset.
 ```
 @inproceedings{commonvoice:2020,
  author = {Ardila, R. and Branson, M. and Davis, K. and Henretty, M. and Kohler, M. and Meyer, J. and Morais, R. and Saunders, L. and Tyers, F. M. and Weber, G.},
  title = {Common Voice: A Massively-Multilingual Speech Corpus},
  booktitle = {Proceedings of the 12th Conference on Language Resources and Evaluation (LREC 2020)},
  pages = {4211--4215},
  year = 2020
}
 ```

The grapheme-to-phoneme conversion was done automatically with [`g2p_en`](https://github.com/Kyubyong/g2p).
```
@misc{g2pE2019,
  author = {Park, Kyubyong & Kim, Jongseok},
  title = {g2pE},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Kyubyong/g2p}}
}
```

## Alignments for Mandarin Speech datasets

This repository contains phone- and word-level alignments for multiple Mandarin Chinese speech datasets, including MagicData (~755 hours), Aishell-1 (~150 hours), STCMDS (~100 hours), Datatang (~200 hours), THCHS-30 (~30 hours) and PrimeWords (~100 hours).

### Textgrids
You can download all textgrids [here](https://drive.google.com/drive/folders/1IF0WB5-8VXfaENtE4r5rehHHK8YFe61S?usp=sharing). The forced alignment was done with `charsiu_forced_aligner` using model `charsiu/zh_xlsr_fc_10ms`. Only Praat textgrid files are distributed. Sentences with Englist letters and numbers were all removed. Misaligned files were also discarded. 

The grapheme-to-phoneme conversion was done automatically with [`g2pM`](https://github.com/kakaobrain/g2pM).
```
@article{park2020g2pm,
 author={Park, Kyubyong and Lee, Seanie},
 title = {A Neural Grapheme-to-Phoneme Conversion Package for Mandarin Chinese Based on a New Open Benchmark Dataset
},
 journal={Proc. Interspeech 2020},
 url = {https://arxiv.org/abs/2004.03136},
 year = {2020}
}
```




### Audio data
The original audio data can be downloaded via OpenSLR. All filenames are matched. Please also cite the original datasets. 

Aishell-1 (~150 hours): https://openslr.org/33/
```
@inproceedings{aishell_2017,
  title={AIShell-1: An Open-Source Mandarin Speech Corpus and A Speech Recognition Baseline},
  author={Hui Bu, Jiayu Du, Xingyu Na, Bengu Wu, Hao Zheng},
  booktitle={Oriental COCOSDA 2017},
  pages={Submitted},
  year={2017}
}
```

MagicData (~755 hours): https://openslr.org/68/
```
Please cite the corpus as "Magic Data Technology Co., Ltd., "http://www.imagicdatatech.com/index.php/home/dataopensource/data_info/id/101", 05/2019".
```

Datatang (~200 hours): https://openslr.org/62/
```
Please cite the corpus as “aidatatang_200zh, a free Chinese Mandarin speech corpus by Beijing DataTang Technology Co., Ltd ( www.datatang.com )”.
```
STCMDS (~100 hours): https://openslr.org/38/
```
Please cite the data as “ST-CMDS-20170001_1, Free ST Chinese Mandarin Corpus”.
```
PrimeWords (~100 hours): https://openslr.org/47/
```
  @misc{primewords_201801,
    title={Primewords Chinese Corpus Set 1},
    author={Primewords Information Technology Co., Ltd.},
    year={2018},
    note={\url{https://www.primewords.cn}}
    }
  
```
THCHS-30 (~30 hours): https://openslr.org/18/
```
@misc{THCHS30_2015,
  title={THCHS-30 : A Free Chinese Speech Corpus},
  author={Dong Wang, Xuewei Zhang, Zhiyong Zhang},
  year={2015},
  url={http://arxiv.org/abs/1512.01882}
}
```
