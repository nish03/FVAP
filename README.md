# **FVAP**

A straightforward training system to improve the algorithmic fairness of visual attribute prediction networks

![Method Overview](assets/title_image.png)

> [Enhancing **F**airness of **V**isual **A**ttribute **P**redictors](https://arxiv.org/abs/2207.05727), <br>
> [_Tobias Hänel_](https://arxiv.org/search/cs?searchtype=author&query=Hänel%2C+Tobias), [_Nishant Kumar_](https://arxiv.org/search/cs?searchtype=author&query=Kumar%2C+Nishant), [_Dmitrij Schlesinger_](https://arxiv.org/search/cs?searchtype=author&query=Schlesinger%2C+Dmitrij), [_Mengze Li_](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+Mengze), [_Erdem Ünal_](https://arxiv.org/search/cs?searchtype=author&query=Ünal%2C+Erdem), [_Abouzar Eslami_](https://arxiv.org/search/cs?searchtype=author&query=Eslami%2C+Abouzar), [_Stefan Gumhold_](https://arxiv.org/search/cs?searchtype=author&query=Gumhold%2C+Stefan), <br>
> *ACCV2022 ([arXiv:2207.05727](https://arxiv.org/abs/2207.05727)*)

## Abstract
Our new training procedure improves the fairness of image classification models w.r.t. to sensitive attributes such as
gender, age, and ethnicity. We add a weighted fairness loss to the standard cross-entropy loss during mini-batch
gradient descent. It estimates the fairness of the model’s  predictions based on the sensitive attributes and the
predicted and ground-truth target attributes from the samples within each batch.

## News
* (2022/09/16) The ACCV2022 program chairs accepted our paper

## Requirements
The code base is designed to be used on Linux/Mac OS.

## Installation (Linux)
* Clone the repository to your desired location of the project root directory (`$PROJ_ROOT`)<br> 
``$ export PROJ_ROOT=$PWD``
* Installing the python virtual environment via virtualenv and the required PyPI libraries
``$ pip install -U virtualenv``
``$ python3 -m virtualenv .env``
``$ source .env/bin/activate``
``$ pip install -r requirements.txt``

## Data preparation
Each supported dataset will be stored an individual folder in the `datasets` directory in the project root (destination of `git clone`).
* **UTKFace** can be obtained from the official [webpage](https://susanqq.github.io/UTKFace/). Select the [Aligned&Cropped Faces](https://drive.google.com/drive/folders/0BxYys69jI14kU0I1YUQyY1ZDRUE?resourcekey=0-01Pth1hq20K4kuGVkp3oBw), download `UTKFace.tar.gz` and extract the contained images to `datasets/UTKFace` in the project root directory (where you cloned this repository to).
* **CelebA**
* **SIIM-ISIC Melanoma Classification**

## Training

## Citation
```
@article{haenel2022fvap,
  title = {Enhancing Fairness of Visual Attribute Predictors},
  author = {Hänel, Tobias and Kumar, Nishant and Schlesinger, Dmitrij and Li, Mengze and Ünal, Erdem and Eslami, Abouzar and Gumhold, Stefan},
  doi = {10.48550/ARXIV.2207.05727},
  url = {https://arxiv.org/abs/2207.05727},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Probability (math.PR), FOS: Computer and information sciences, FOS: Computer and information sciences, FOS: Mathematics, FOS: Mathematics},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
```