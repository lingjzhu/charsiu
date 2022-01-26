## Charsiu: A transformer-based phonetic aligner [[arXiv]](https://arxiv.org/abs/2110.03876)

**[UPDATE]** Mandarin models are now available on the model hub. 

**Note.** The aligner is under active development. New functions, new languages and detailed documentation will be added soon! Give us a star if you like our project!

### Intro
**Charsiu** is a phonetic alignment tool, which can:
- recognise phonemes in a given audio file
- perform forced alignment using phone transcriptions created in the previous step or provided by the user.
- directly predict the phone-to-audio alignment from audio (text-independent alignment)  

Pretrained models are available at the 🤗 *HuggingFace* model hub: https://huggingface.co/charsiu.

**Fun fact**: Char Siu is one of the most representative dishes of Cantonese cuisine 🍲 (see [wiki](https://en.wikipedia.org/wiki/Char_siu)). 


### Tutorial 
**[!NEW]** A step-by-step tutorial for linguists: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lingjzhu/charsiu/blob/development/charsiu_tutorial.ipynb)

You can directly run our model in the cloud via Google Colab!  
 - Forced alignment:   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lingjzhu/charsiu/blob/development/charsiu_forced_alignment_demo.ipynb)  
 - Textless alignment: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lingjzhu/charsiu/blob/development/charsiu_textless_demo.ipynb)  

### Usage
```
git clone  https://github.com/lingjzhu/charsiu
cd charsiu
```
#### Forced alignment
```Python
from Charsiu import charsiu_forced_aligner

# initialize model
charsiu = charsiu_forced_aligner(aligner='charsiu/en_w2v2_fc_10ms')
# perform forced alignment
alignment = charsiu.align(audio='./local/SA1.WAV',
                          text='She had your dark suit in greasy wash water all year.')
# perform forced alignment and save the output as a textgrid file
charsiu.serve(audio='./local/SA1.WAV',
              text='She had your dark suit in greasy wash water all year.',
              save_to='./local/SA1.TextGrid')


# Chinese
charsiu = charsiu_forced_aligner(aligner='charsiu/zh_w2v2_tiny_fc_10ms',lang='zh')
charsiu.align(audio='./local/SSB00050015_16k.wav',text='经广州日报报道后成为了社会热点。')
charsiu.serve(audio='./local/SSB00050015_16k.wav', text='经广州日报报道后成为了社会热点。',
              save_to='./local/SSB00050015.TextGrid')
```


#### Textless alignment
```Python
from Charsiu import charsiu_predictive_aligner
# English
# initialize a model
charsiu = charsiu_predictive_aligner(aligner='charsiu/en_w2v2_fc_10ms')
# perform textless alignment
alignment = charsiu.align(audio='./local/SA1.WAV')
# Or
# perform textless alignment and output the results to a textgrid file
charsiu.serve(audio='./local/SA1.WAV', save_to='./local/SA1.TextGrid')


# Chinese
charsiu = charsiu_predictive_aligner(aligner='charsiu/zh_xlsr_fc_10ms',lang='zh')

charsiu.align(audio='./local/SSB16240001_16k.wav')
# Or
charsiu.serve(audio='./local/SSB16240001_16k.wav', save_to='./local/SSB16240001.TextGrid')
```
### Development plan

 - Package  

|     Items          | Progress |
|:------------------:|:--------:|
|  Documentation     | Nov 2021 |    
|  Textgrid support  |     √    |
| Word Segmentation  |     √    |
| Model compression  |   TBD    |
|  IPA support       |   TBD    |

 - Multilingual support

|      Language      | Progress |
|:------------------:|:--------:|
| English (American) |     √    |
|  Mandarin Chinese  |     √    |
|       German       | Jan 2022 |
|       Spanish      | Feb 2022 |
|  English (British) |    TBD   |
|    Cantonese       |    TBD   |
|    AAVE            |    TBD   |





### Dependencies
pytorch  
transformers  
datasets  
librosa  
g2pe  
praatio  
g2pM


### Training
Coming soon!

Note.Training code is in `experiments/`. Those were original research code for training the model. They still need to be reorganized. 

### Finetuning
Coming soon!

### Attribution and Citation
For now, you can cite this tool as:

```
@article{zhu2019charsiu,
  title={Phone-to-audio alignment without text: A Semi-supervised Approach},
  author={Zhu, Jian and Zhang, Cong and Jurgens, David},
  journal={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2022}
 }
```
Or


To share a direct web link: https://github.com/lingjzhu/charsiu/.

### References
[Transformers](https://huggingface.co/transformers/)  
[s3prl](https://github.com/s3prl/s3prl)  
[Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/)


### Disclaimer

This tool is a beta version and is still under active development. It may have bugs and quirks, alongside the difficulties and provisos which are described throughout the documentation. 
This tool is distributed under MIT license. Please see [license](https://github.com/lingjzhu/charsiu/blob/main/LICENSE) for details. 

By using this tool, you acknowledge:

* That you understand that this tool does not produce perfect camera-ready data, and that all results should be hand-checked for sanity's sake, or at the very least, noise should be taken into account.

* That you understand that this tool is a work in progress which may contain bugs.  Future versions will be released, and bug fixes (and additions) will not necessarily be advertised.

* That this tool may break with future updates of the various dependencies, and that the authors are not required to repair the package when that happens.

* That you understand that the authors are not required or necessarily available to fix bugs which are encountered (although you're welcome to submit bug reports to Jian Zhu (lingjzhu@umich.edu), if needed), nor to modify the tool to your needs.

* That you will acknowledge the authors of the tool if you use, modify, fork, or re-use the code in your future work.  

* That rather than re-distributing this tool to other researchers, you will instead advise them to download the latest version from the website.

... and, most importantly:

* That neither the authors, our collaborators, nor the the University of Michigan or any related universities on the whole, are responsible for the results obtained from the proper or improper usage of the tool, and that the tool is provided as-is, as a service to our fellow linguists.

All that said, thanks for using our tool, and we hope it works wonderfully for you!

### Support or Contact
Please contact Jian Zhu ([lingjzhu@umich.edu](lingjzhu@umich.edu)) for technical support.  
Contact Cong Zhang ([cong.zhang@ru.nl](cong.zhang@ru.nl)) if you would like to receive more instructions on how to use the package.



