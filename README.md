## Charsiu: A transformer-based phonetic aligner [[arXiv]](https://arxiv.org/abs/2110.03876)

**Note.** This is a preview version. The aligner is under active development. New functions, new languages and detailed documentation will be added soon!

### Intro
**Charsiu** is a phonetic alignment tool, which can:
- recognise phonemes in a given audio file
- perform forced alignment using phone transcriptions created in the previous step or provided by the user.
- directly predict the phone-to-audio alignment from audio (text-independent alignment)

**Fun fact**: Char Siu is one of the most representative dishes of Cantonese cuisine 🍲 (see [wiki](https://en.wikipedia.org/wiki/Char_siu)). 


### Tutorial (In progress)
You can directly run our model in the cloud via Google Colab!  
 - Forced alignment:   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lingjzhu/charsiu/blob/development/charsiu_forced_alignment_demo.ipynb)  
 - Textless alignment: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lingjzhu/charsiu/blob/development/charsiu_textless_demo.ipynb)  



### Development plan

 - Package  

|     Items          | Progress |
|:------------------:|:--------:|
|  Documentation     | Nov 2021 |    
|  Textgrid support  |     √    |
| Model compression  |   TBD    |
|  IPA support       |   TBD    |

 - Multilingual support

|      Language      | Progress |
|:------------------:|:--------:|
| English (American) |     √    |
|  Mandarin Chinese  | Nov 2021 |
|       Spanish      | Jan 2022 |
|  English (British) |    TBD   |
|    Cantonese       |    TBD   |
|    AAVE            |    TBD   |

### Pretrained models
Our pretrained models are available at the *HuggingFace* model hub: https://huggingface.co/charsiu.


### Dependencies
pytorch  
transformers  
datasets  
librosa  
g2pe  
praatio

Training code is in `experiments/`. Those were original research code for training the model. They still need to be reorganized. 

### Training
Coming soon!

### Finetuning
Coming soon!

### Attribution and Citation
For now, you can cite this tool as:

```
@article{zhu2019charsiu,
  title={Phone-to-audio alignment without text: A Semi-supervised Approach},
  author={Zhu, Jian and Zhang, Cong and Jurgens, David},
  journal={arXiv preprint arXiv:https://arxiv.org/abs/2110.03876},
  year={2021}
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



