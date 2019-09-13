# Weakly supervised classification of aortic valve malformations using unlabeled cardiac MRI sequences
[![DOI](https://zenodo.org/badge/134775347.svg)](https://zenodo.org/badge/latestdoi/134775347)

This is the code release for our paper in Nature Communications. 
https://www.nature.com/articles/s41467-019-11012-3


## Content
* Overview
* Repo Content
* System Requirements
* Installation Guide
* How to Use

## Overview
Bicuspid Aortic Valve (BAV) is the most common congenital malformation of the heart, occurring in 0.5-2% of the general population. We developed a weakly supervised deep learning model for BAV classification using up to 4,000 unlabeled cardiac MRI sequences.

## Repo Content
* *notebooks/* - Weak supervision notebooks.
* *scripts/* - Launch scripts for Supervised model, Weakly-supervised model and Expert-Weakly supervised model
* *ukb/* - the python code and config files needed for training the various models.
* *data.zip* - data provided to run our pipeline (this is all synthetic data, no real patient data was used)

## I. System Requirements

### Hardware Requirements
Weakly supervised MRI classification needs a computer with the following minimum specs:

* CPU : A cores, B Ghz/core
* GPU : CUDA is compatible with almost all NVidia models from 2006, but a minimum of gtx 1050ti, 1060 and above are required.
* RAM : A minimum of 16 GB RAM is required.

All of our runtime estimates are generated from a computer with the following specs:

* CPU : Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60GHz. 56 CPUs in total.
* GPU : NVidia Tesla P100-PCIE-16GB
* RAM : 503 GB.

### Software Requirements
#### OS Requirements
This package is supported for *Linux* operating systems and has been tested on the following system:

* Linux - GPU/CPU
* Mac OSX - CPU only


Before installing the package, be sure to have the following software installed on your system:

* Python 3.6.4
* Several python packages which can be installed via ```pip``` (see below)

#### Installing Anaconda Distribution of Python
Download the Anaconda installer and install in terminal, detailed instructions can be found in:
```
https://conda.io/docs/user-guide/install/linux.html
```

#### Setting Up a conda Environment
If you plan on creating a standalone environment (conda or virtualenv) please take note of the following. Since matplotlib is imported a lot, we suggest creating a conda environment (as you can install a framework build of python easily and use the package) rather than a virtualenv (which as of this writing installs a non-framework version of python and causes a lot of scripts to crash).
```
conda create -n myEnv python=3.6.4 pip
```
Once you have successfully created your conda environment, be sure to activate it:
```
source activate myEnv
```
Now that your environment is active run the following commands to install all requirements:
* Make Python in the environment a framework build:
```
conda install python.app
```

#### Installing Python package dependencies
To ensure that all python package dependacies are installed, run the following command:

```
pip install -r requirements.txt --find-links=http://download.pytorch.org/whl/torch-0.3.1-cp27-none-macosx_10_6_x86_64.whl --trusted-host download.pytorch.org
```

Package List:
```
backports.functools-lru-cache==1.5
certifi==2018.1.18
cffi==1.11.5
chardet==3.0.4
cycler==0.10.0
decorator==4.2.1
dominate==2.3.1
idna==2.7
imageio==2.3.0
kiwisolver==1.0.1
matplotlib==2.2.0
networkx==2.1
numpy==1.14.2
opencv-contrib-python-headless==3.4.3.18
pandas==0.22.0
Pillow==5.0.0
pycparser==2.18
pyparsing==2.2.0
python-dateutil==2.7.0
pytz==2018.3
PyWavelets==0.5.2
PyYAML==3.12
requests==2.19.1
scikit-image==0.13.1
scikit-learn==0.19.1
scipy==1.0.0
seaborn==0.8.1
six==1.11.0
tabulate==0.8.2
torch==0.4.0
torchvision==0.2.0
urllib3==1.23
```

## II. Installation Guide
To install this package, clone our repo on your system.

Once it has been downloaded, be sure to run the following command:
```
unzip data.zip
```
so that you may utilize our provided data.

## III. How to Use
We have provided various scripts to run our model. These scripts are located in the *scripts/* directory. To execute our example script run the following command:
```
./scripts/Supervised.sh
```

This will launch 5 different jobs taking about 7 GB of GPU memory. In addition, this entire run will take roughly 3.5 hrs. The expected output is (located in Experiments/out/seed_x.out, e.g. Supervised/out/seed_0.out):

```
========================================
Scores
========================================
Pos. class accuracy: 75.0
Neg. class accuracy: 96.2
----------------------------------------
AUC:                 96.9
PRC:                 48.8
NDCG:                78.4
----------------------------------------
Precision:           42.9
Recall:              75.0
F1:                  54.5
----------------------------------------
TP: 6 | FP: 8 | TN: 200 | FN: 2
========================================

```

After all 5 different jobs are finished, another script is provided for generating predictions on the example DEV/TEST set, and collecting the ensemble of TEST results. To execute the script, simply run the following command:
```
./scripts/predict_Supervised.sh
```
The generated predictions will be located in Experiments/predictions, e.g. Supervised/predictions. And the collected results ensemble would be in Experiments/predictions/results_test/ensemble, e.g. Supervised/predictions/results_test/ensemble. The terminal output would look like this:
```
   Experiment SEED_0 SEED_14 SEED_57 SEED_123 SEED_1234 AVERAGE   STD MEDIAN     MV
0     Pos.Acc   0.00    0.00    0.00     0.00      0.00    0.00  0.00   0.00   0.00
1     Neg.Acc  93.10   94.25   89.66    91.95     96.55   93.10  2.30  98.85  94.25
2   Precision   0.00    0.00    0.00     0.00      0.00    0.00  0.00   0.00   0.00
3      Recall   0.00    0.00    0.00     0.00      0.00    0.00  0.00   0.00   0.00
4          F1   0.00    0.00    0.00     0.00      0.00    0.00  0.00   0.00   0.00
5         ROC  80.46   76.25   60.54    65.90     63.98   69.43  7.61  72.80  42.53
6         PRC   7.50    7.30    3.95     4.56      5.22    5.70  1.44   5.42   1.67
7        NDCG  33.73   33.60   27.05    28.06     31.94   30.88  2.80  30.72  26.98
8          TP   0.00    0.00    0.00     0.00      0.00    0.00  0.00   0.00   0.00
9          FP   6.00    5.00    9.00     7.00      3.00    6.00  2.00   1.00   5.00
10         TN  81.00   82.00   78.00    80.00     84.00   81.00  2.00  86.00  82.00
11         FN   3.00    3.00    3.00     3.00      3.00    3.00  0.00   3.00   3.00
```
