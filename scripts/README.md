# Instructions

## Basics
In order to start up, change the arguments in `scaleup_base.sh` and `SupervisedAugmentation.sh`:
```
REPO="/dfs/scratch0/kexiao/ukb-cardiac-mri/ukb"
TRAIN="/lfs/1/heartmri/coral32"
DEV="/lfs/1/heartmri/dev32"
SEMI="/lfs/1/heartmri/semi32"
SEMI_CSV="labels_train.csv"
```
to the data and code location on the machine.

## For launching Weak Supervised model training
```
./WeakSupervised.sh
```

## For launching Weak Supervised model with augmentation training
```
./WeakSupervisedAugmentation.sh
```

## For launching Expert/Weak Supervised model training
```
./ExpertWeakSupervised.sh
```

## For launching Expert/Weak Supervised model with augmentation training
```
./ExpertWeakSupervisedAugmentation.sh
```

## For launching Supervised model with augmentation training
```
./SupervisedAugmentation.sh
```

## Other options
### use first argument to launch a single seed model training. 
E.g. 
```
./WeakSupervised.sh 14
```
for launching the Weak Supervised model training with one seed `14` instead of default 5 seeds.

### change the N_SAMPLES="100" to change the scale of training
E.g.
```
use N_SAMPLES="1200" to train on 1200 samples instead of default 100 samples.
```
