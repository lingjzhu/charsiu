## Alignments for several Mandarin Speech datasets

This repository contains phone- and word-level alignments for multiple Mandarin Chinese speech datasets, including MagicData (~755 hours), Aishell-1 (~150 hours), STCMDS (~100 hours), Datatang (~200 hours), THCHS-30 (~20 hours) and PrimeWords (~100 hours).

### Textgrids
You can download all textgrids [here](https://drive.google.com/drive/folders/1IF0WB5-8VXfaENtE4r5rehHHK8YFe61S?usp=sharing). The forced alignment was done with [Charsiu Forced Aligner](https://github.com/lingjzhu/charsiu) using model `charsiu/zh_xlsr_fc_10ms`. Only Praat textgrid files are distributed. Sentences with Englist letters and numbers were all removed. Misaligned files were also discarded. Forced alignment does not generate perfect alignments. **Use at you own discrection**.

Please cite this if you use these alignments in your research projects.
```
@article{zhu2019charsiu,
  title={Phone-to-audio alignment without text: A Semi-supervised Approach},
  author={Zhu, Jian and Zhang, Cong and Jurgens, David},
  journal={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2022},
  url=https://arxiv.org/abs/2110.03876,
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
