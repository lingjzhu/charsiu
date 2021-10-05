# Charsiu: A transformer-based phonetic aligner

### Intro
**Charsiu** is a phonetic alignment tool, which can:
- automatically recognise the text and phonemes in a given audio file
- automatically align the phonemes and generate time stamps for the phonemes using text transcription created by the previous step or provided by the user.


#### Fun fact
Char Siu is one of the most representative dishes of Cantonese cuisine ([wiki](https://en.wikipedia.org/wiki/Char_siu)). 


### Demo


### Tutorial
Coming soon!


### Development plan
|      Language      | Progress |
|:------------------:|:--------:|
| English (American) |     √    |
|  Mandarin Chinese  | Nov 2021 |
|       Spanish      | Dec 2021 |
|  English (British) |    TBD   |
|    Cantonese       |    TBD   |


### Pretrained models
Our pretrained models are availble at the *HuggingFace* model hub: https://huggingface.co/charsiu.


### Dependencies
pytorch  
transformers  
datasets  
librosa
g2pe  

### Training
Coming soon!

### Finetuning
Coming soon!


### Attribution and Citation
For now, you can cite this tool as:

>@article{zhu2019cnn,
>  title={Semi-supervised Learning of phone-to-audio alignment},
>  author={Zhu, Jian and Zhang, Cong and Jurgens, David},
>  journal={arXiv preprint arXiv:????????????????????},
>  year={2021}
> }
Or


To share a direct web link: https://github.com/lingjzhu/charsiu/.

### References
[Transformers](https://huggingface.co/transformers/)  
[s3prl](https://github.com/s3prl/s3prl)  
[Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/)


### Disclaimer

This tool is a beta version and is still under active development. It may have bugs and quirks, alongside the difficulties and provisos which are described throughout the documentation. 
This tool is distributed under MIT liscence. Please see [license](https://github.com/lingjzhu/charsiu/blob/main/LICENSE) for details. 

By using this tool, you acknowledge:

That you understand that this tool does not produce perfect camera-ready data, and that all results should be hand-checked for sanity's sake, or at the very least, noise should be taken into account.

That you understand that this tool is a work in progress which may contain bugs. Future versions will be released, and bug fixes (and additions) will not necessarily be advertised.

That this tool may break with future updates of the various dependencies, and that the authors are not required to repair the package when that happens.

That you understand that the authors are not required or necessarily available to fix bugs which are encountered (although you're welcome to submit bug reports to Jian Zhu (lingjzhu@umich.edu), if needed), nor to modify the tool to your needs.

That you will acknowledge the authors of the tool if you use, modify, fork, or re-use the code in your future work.

That rather than re-distributing this tool to other researchers, you will instead advise them to download the latest version from the website.

... and, most importantly:

That neither the authors, our collaborators, nor the the University of Michigan on the whole, are responsible for the results obtained from the proper or improper usage of the tool, and that the tool is provided as-is, as a service to our fellow linguists.

All that said, thanks for using our tool, and we hope it works wonderfully for you!

### Support or Contact
Please contact Jian Zhu (lingjzhu@umich.edu) for technical support. Contact Cong Zhang (cong.zhang@ru.nl) if you would like to receive more instructions on how to use the package.



