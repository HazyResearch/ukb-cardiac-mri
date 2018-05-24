# Weakly Supervised MRI Classification

## I. Environment Setup
If you plan on creating a standalone environment (conda or virtualenv) please take note of the following. Since matplotlib is imported a lot, we suggest creating a conda environment (as you can install a framework build of python easily and use the package) rather than a virtualenv (which as of this writing installs a non-framework version of python and causes a lot of scripts to crash).
```
conda create -n myEnv python=2.7.* pip
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
**NOTE**: to use the framework build of python, you need to use **pythonw** instead of python
```
pythonw execute_script.py
```
* Install all Python package dependencies:
```
pip install -r requirements.txt --find-links=http://download.pytorch.org/whl/torch-0.3.1-cp27-none-macosx_10_6_x86_64.whl --trusted-host download.pytorch.org
```

## II. Quick Start
Create a sample dataset of 15x32x32 with class balance 0.90/0.10 and train a simple LeNet+RNN model. NOTE: This randomly splits dev into a stratified dev/test sample. Change the random seed to get a different split.

```
mkdir data/mri
mkdir data/results
pythonw create_synthetic_data.py -o data/mri -n 200 --full -I mri -F 30 -D 32 -P 0.10
pythonw train.py --train data/mri/train/ --dev data/mri/dev/ -c configs/ukbb/lenet1_rnn.json -B 4 -N 5 -E 10 -F 30 --outdir data/results/ --update_freq 1 --stratify_dev --report --seed 1234
```
You can also train on real data from the `CIFAR10` dataset

## III. Training Models

```
pythonw train.py --train mri/train/ --dev mri/dev/ -c configs/ukbb/lenet1_rnn.json -B 4 -N 5 -E 10 -F 30 --outdir results/ update_freq 1 --stratify_dev --report --seed 1234
```

All models and results are checkpointed into `outdir`

### Train/Dev/Test Split Definitions
Splits are defined as a CSV of tuples of `SUBJECT ID` and `BAV PROBA`

	100001,0.9
	100002,0.2

### Model Configurations
Model configurations are found in `configs/<DATASET>/<MODEL>.json`



## IV. Segmenting and Localizing the Aortic Valve

### Preprocessing UKBB MRI Data
All models above assume 32x32 - 48x48 input image size. Localizing MRIs to the just include the aortic valve can be done heuristically and acheive ~98% accuracy.

This will segment all patients:

```
pythonw segment.py -i <SHERLOCK>/BAV/data/original/ -o <SHERLOCK>/BAV/data/no_mask_crop_32/ -n 15000 -D 32 --mask none --pooling none --format npy
```
Optionally, you can provided a `--cohort` consisting of subject IDs to just create a subset of segmented images.


## V. Create Synthetic Training Datasets

We can use the same command as in the quickstart part.
Create `200` random MRI sequence data of shape `30 x 32 x 32` with class balance `75/25`

```
python create_synthetic_data.py -o <OUTDIR> -n 200 -I mri -D 32 -P 0.25
```
