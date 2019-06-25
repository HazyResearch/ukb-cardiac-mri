from __future__ import print_function, division
import os
import sys
import logging
import numpy as np
import pandas as pd
from skimage.color import grey2rgb
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from collections import OrderedDict
from sklearn.preprocessing import OneHotEncoder


logger = logging.getLogger(__name__)

seriesMap = {
    0   :   "flow_250_tp_AoV_bh_ePAT@c_MAG",
    1   :   "flow_250_tp_AoV_bh_ePAT@c_P",
    2   :   "flow_250_tp_AoV_bh_ePAT@c"
}

AGE_MEAN    = 55.360334580673566
AGE_STD     = 7.569607026616798

class UKBBCardiacMRI(Dataset):
    """
    UK Biobank cardiac MRI dataset
    Load Numpy MRI sequence tensors with shape:
        num_frames X width X height

    TODO: Transformations are custom and are applied to each frame

    """
    def __init__(self, csv_data, root_dir, series=0, N=30, image_type='grey',
                 preprocess=None, augmentation=None, postprocess=None,
                 rebalance=False, threshold=0.5, seed=4321,
                 sample=False, sample_type=0, sample_split=0.5, n_samples=100,
                 pos_samples=0, neg_samples=0, frame_label=False, rebalance_strategy="oversample",
                 semi=False, semi_dir=None, semi_csv=None):
        # either load from CSV or just use the provided pandas dataframe
        if frame_label:
            csv_data = "{}/labels_frame.csv".format(root_dir)
        self.labels = pd.read_csv(csv_data) if type(csv_data) is str else csv_data
        self.root_dir = root_dir
        self.series = series
        self.preprocess = preprocess
        self.augment = augmentation
        self.postprocess = postprocess
        self.image_type = image_type
        self.N = N
        self.frame_label = frame_label
        self.rebalance_strategy = rebalance_strategy
        self.semi = semi
        self.semi_dir = semi_dir
        self.semi_csv = semi_csv
        np.random.seed(seed)

        # remove any instances with label = 0.5
        #df = [{"ID":x.ID,"LABEL":x.LABEL} for x in list(self.labels.itertuples()) if x.LABEL != 0.5]
        #self.labels = pd.DataFrame(df)

        # Sampling is computed first, then rebalancing if wanted :)
        if sample:
            pids, ys = zip(*[(x.ID,x.LABEL) for x in list(self.labels.itertuples())])
            pids = np.array(pids)
            ys = np.array(ys)
            Content = OrderedDict([(str(key), np.array(getattr(self.labels, key))) for key in self.labels.columns if key != 'ID' and key != 'LABEL'])

            if (sample_type == 0):
                logger.info("Randomly Sampling dataset...\n" +
                            "\tNum Samples  = {}\n".format(n_samples))
                self.random_sample(pids, ys, n_samples, Content)
            elif (sample_type == 1):
                logger.info("Threshold Random Sampling dataset...\n" +
                            "\tSample Split = {}\n".format(sample_split) +
                            "\tThreshold    = {}\n".format(threshold) +
                            "\tNum Samples  = {}\n".format(n_samples))
                self.threshold_random_sample(pids, ys, sample_split, threshold, n_samples, Content)
            elif (sample_type == 2):
                logger.info("Top/Bottom Sampling dataset...\n" +
                            "\tSample Split = {}\n".format(sample_split) +
                            "\tNum Samples  = {}\n".format(n_samples))
                self.top_bottom_sample(pids, ys, sample_split, n_samples, Content)
            elif (sample_type == 3):
                logger.info("Random Subset Sampling dataset...\n" +
                            "\tThreshold     = {}\n".format(threshold) +
                            "\tPos Samples   = {}\n".format(pos_samples) +
                            "\tNeg Samples   = {}\n".format(neg_samples) +
                            "\tTotal Samples = {}\n".format(pos_samples + neg_samples))
                self.random_subset_sampling(pids, ys, threshold, pos_samples, neg_samples, Content)

        # append hard labels to create semi-supervised labels DataFrame
        if semi:
            semi_labels = pd.read_csv("{}/{}".format(self.semi_dir, self.semi_csv))
            self.labels = self.labels.append(semi_labels, ignore_index=True).ix[:,semi_labels.columns]


        # resample to create balanced training set?
        # HACK: assumes binary classes
        if rebalance:
            logger.info("Rebalancing dataset... {} b={}".format(self.rebalance_strategy, threshold))

            pids, ys = zip(*[(x.ID,x.LABEL) for x in list(self.labels.itertuples())])
            pids = np.array(pids)
            ys = np.array(ys)
            t = [i for i,v in enumerate(ys) if v > threshold]
            f = [i for i,v in enumerate(ys) if v <= threshold]
            logger.info("True:{:>4}  False:{:>4}".format(len(t), len(f)))

            # oversample minority class OR undersample majority class
            minority_class, majority_class = (t,f) if len(t) < len(f) else (f,t)

            if self.rebalance_strategy == "oversample":
                minority_class = np.random.choice(minority_class, len(majority_class), replace=True)

            elif self.rebalance_strategy == "undersample":
                majority_class = np.random.choice(majority_class, len(minority_class), replace=True)

            #logger.info("Minority:{:>4}  Majority:{:>4}".format(len(minority_class), len(majority_class)))
            df = []
            for pid,label in zip(pids[minority_class], ys[minority_class]):
                df.append(self.labels.loc[self.labels.ID==pid].to_dict('records', into=OrderedDict)[0])
                #df.append({"ID":id, "LABEL":label})

            for pid,label in zip(pids[majority_class], ys[majority_class]):
                df.append(self.labels.loc[self.labels.ID==pid].to_dict('records', into=OrderedDict)[0])
                #df.append({"ID":id, "LABEL":label})

            self.labels = pd.DataFrame(df)

            # sanity check
            pids, ys = zip(*[(x.ID, x.LABEL) for x in list(self.labels.itertuples())])
            pids = np.array(pids)
            ys = np.array(ys)

            t = [i for i, v in enumerate(ys) if v > threshold]
            f = [i for i, v in enumerate(ys) if v <= threshold]
            logger.info("True:{:>4}  False:{:>4}".format(len(t), len(f)))

    def random_sample(self, pids, ys, x, Content):
        """
        Randomly sample x patients.

        Params
        ------
        pids    :   np.array
            - patient id array

        ys      :   np.array
            - patient label array

        x       :   int
            - number of patients to sample

        Return
        ------
        None
            - set self.labels = pd.DataFrame(something)
        """
        indexes = np.arange(len(pids))
        np.random.shuffle(indexes)

        df = []
        for idx, (id, label) in enumerate(zip(pids[indexes[:x]], ys[indexes[:x]])):
            row = OrderedDict([("ID",id), ("LABEL",label)])
            for key in Content:
                row.update({key : Content[key][indexes[idx]]})
            df.append(row)

        self.labels = pd.DataFrame(df)

    def threshold_random_sample(self, pids, ys, split, threshold, x, Content):
        """
        Randomly sample x patients based on a given threshold.

        Params
        ------
        pids    :   np.array
            - patient id array

        ys      :   np.array
            - patient label array

        x       :   int
            - number of patients to sample

        Return
        ------
        None
            - set self.labels = pd.DataFrame(something)
        """
        # Determine the split of patients
        pos_count = int(round(x*split))
        neg_count = x - pos_count

        # Separate patients based on threshold
        p = [i for i,v in enumerate(ys) if v > threshold]
        n = [i for i,v in enumerate(ys) if v <= threshold]
        logger.info("Class Distribution :\n" +
                    "\tPossitive Class : {:>4}\n\tNegative Class  : {:>4}\n".format(len(p), len(n)))

        np.random.shuffle(p)
        np.random.shuffle(n)

        logger.info("Class Selection Count :\n" +
                    "\tPossitive Class : {:>4}\n\tNegative Class  : {:>4}\n".format(pos_count, neg_count))
        df = []
        for idx, (id,label) in enumerate(zip(pids[p[:pos_count]], ys[p[:pos_count]])):
            row = OrderedDict([("ID",id), ("LABEL",label)])
            for key in Content:
                row.update({key : Content[key][p[idx]]})
            df.append(row)

        for idx, (id,label) in enumerate(zip(pids[n[:neg_count]], ys[n[:neg_count]])):
            row = OrderedDict([("ID",id), ("LABEL",label)])
            for key in Content:
                row.update({key : Content[key][n[idx]]})
            df.append(row)

        self.labels = pd.DataFrame(df)

    def top_bottom_sample(self, pids, ys, split, x, Content):
        """
        Sample x patients from top and bottom of labels.

        Params
        ------
        pids    :   np.array
            - patient id array

        ys      :   np.array
            - patient label array

        x       :   int
            - number of patients to sample

        Return
        ------
        None
            - set self.labels = pd.DataFrame(something)
        """
        # Determine the split of patients
        pos_count = int(round(x*split))
        neg_count = x - pos_count

        index_sort = np.argsort(ys)
        sorted_ys = ys[index_sort]
        sorted_pids = pids[index_sort]

        logger.info("Class Selection Count :\n" +
                    "\tPossitive Class : {:>4}\n\tNegative Class  : {:>4}\n".format(pos_count, neg_count))
        df = []
        # Get highest probability (highest labels) 'positive' cases
        for idx, (id, label) in enumerate(zip(sorted_pids[-pos_count:], sorted_ys[-pos_count:])):
            row = OrderedDict([("ID",id), ("LABEL",label)])
            for key in Content:
                row.update({key : Content[key][index_sort[-pos_count+idx]]})
            df.append(row)
        # Get lowest probability (lowest labels) 'negative' cases
        for idx, (id, label) in enumerate(zip(sorted_pids[:neg_count], sorted_ys[:neg_count])):
            row = OrderedDict([("ID",id), ("LABEL",label)])
            for key in Content:
                row.update({key : Content[key][index_sort[idx]]})
            df.append(row)

        self.labels = pd.DataFrame(df)

    def random_subset_sampling(self, pids, ys, threshold, pos_cases, neg_cases, Content):
        """
        Randomly sample subsets of cases and non cases.

        Return a set of total_cases = pos_cases + neg_cases

        Params
        ------
        pids    :   np.array
            - patient id array

        ys      :   np.array
            - patient label array

        threshold   :   int
            - threshold to separate cases and non cases

        pos_cases   :   int
            - number of positive cases to select

        neg_cases   :   int
            - number of negative cases to select

        Return
        ------
        None
            - set self.labels = pd.DataFrame(something)
        """
        # Separate patients based on threshold
        p = [i for i, v in enumerate(ys) if v > threshold]
        n = [i for i, v in enumerate(ys) if v <= threshold]
        logger.info("Class Distribution :\n" +
                    "\tPossitive Class : {:>4}\n\tNegative Class  : {:>4}\n".format(len(p), len(n)))

        np.random.shuffle(p)
        np.random.shuffle(n)

        logger.info("Class Selection Count :\n" +
                    "\tPossitive Class : {:>4}\n\tNegative Class  : {:>4}\n".format(pos_cases, neg_cases))
        df = []
        for idx, (id, label) in enumerate(zip(pids[p[:pos_cases]], ys[p[:pos_cases]])):
            row = OrderedDict([("ID", id), ("LABEL", label)])
            for key in Content:
                row.update({key : Content[key][p[idx]]})
            df.append(row)

        for idx, (id, label) in enumerate(zip(pids[n[:neg_cases]], ys[n[:neg_cases]])):
            row = OrderedDict([("ID", id), ("LABEL", label)])
            for key in Content:
                row.update({key : Content[key][n[idx]]})
            df.append(row)

        self.labels = pd.DataFrame(df)


    def summary(self):
        """
        Generate message summarizing data (e.g., counts, class balance)
        Assumes hard labels
        :return:
        """
        return "Instances: {}".format(len(self))

    def load_label(self, idx):
        # most PyTorch operations are only defined over float or doublefloat (32 vs 64bit) tensors
        if self.frame_label:
            label = np.array(self.labels.iloc[idx, 2:8]).astype(float)
        else:
            label = self.labels.iloc[idx, 1]

        return label

    def get_labels(self):
        return [(str(self.labels.iloc[i, 0]), data[1]) for i, data in enumerate(self)]

    def convert_image(self, images):
        try:
            images = np.array(images)
        except Exception as err:
            raise ValueError("image channels are having different shapes. \n ERR| {}".format(err))

        if images.ndim == 4:
            images = np.moveaxis(images, 0, 1)
        elif self.image_type == 'grey':
            images = np.expand_dims(images, axis=1)
        elif self.image_type == 'rgb':
            images = np.moveaxis(grey2rgb(images), -1, 1)

        return images

    def flow_250_MAG(self, pid, rootDir):
        """
        Load the flow_250_tp_AoV_bh_ePAT@c_MAG series for the given patient.

        Params
        ------
        pid :   str
            - patient id

        Return
        ------
        np.array
            - numpy series
        """
        fpath = os.path.join(rootDir, seriesMap[0] + "/" + pid + ".npy")
        series = np.load(fpath).astype(np.float32)

        if self.preprocess:
            # Apply Preprocessing
            series = self.preprocess(series)

        if self.augment:
            # Apply Agumentations
            # raise NotImplemented()
            series = self.augment(series)

        if self.postprocess:
            series = self.postprocess(series)

        # Compute final 1 Ch or 3 ch
        series = self.convert_image(series)

        return series

    def flow_250_other(self, pid, rootDir):
        """
        Load the flow_250_tp_AoV_bh_ePAT@c* series for the given patient.

        Params
        ------
        pid :   str
            - patient id

        series  :   int
            - series map number
                1 : flow_250_tp_AoV_bh_ePAT@c_P
                2 : flow_250_tp_AoV_bh_ePAT@c

        Return
        ------
        np.array
            - numpy series
        """
        fpath = os.path.join(rootDir, seriesMap[self.series] + "/" + pid + ".npy")
        series = np.load(fpath).astype(np.float32)

        if self.postprocess:
            series = self.postprocess(series)

        # Compute final 1 Ch or 3 ch
        series = self.convert_image(series)

        return series

    def flow_250_all(self, pid, rootDir):
        """
        Load ALL flow_250_tp_AoV_bh_ePAT@c* series for the given patient.

        Params
        ------
        pid :   str
            - patient id

        Return
        ------
        np.array
            - numpy series
        """
        # flow_250_tp_AoV_bh_ePAT@c_MAG
        fpath_MAG = os.path.join(rootDir, seriesMap[0] + "/" + pid + ".npy")
        # flow_250_tp_AoV_bh_ePAT@c_P
        fpath_P = os.path.join(rootDir, seriesMap[1] + "/" + pid + ".npy")
        # flow_250_tp_AoV_bh_ePAT@c
        fpath_c = os.path.join(rootDir, seriesMap[2] + "/" + pid + ".npy")

        # most PyTorch operations are only defined over float or doublefloat (32 vs 64bit) tensors
        series_MAG = np.load(fpath_MAG).astype(np.float32)
        series_P = np.load(fpath_P).astype(np.float32)
        series_c = np.load(fpath_c).astype(np.float32)

        if self.preprocess:
            # Apply Preprocessing
            series_MAG, series_P, series_c = self.preprocess(series_MAG, series_P, series_c)

        # if self.augment:
            # Apply Agumentations
            # raise NotImplemented()

        if self.postprocess:
            series_MAG, series_P, series_c = self.postprocess(series_MAG, series_P, series_c)

        series = [series_MAG, series_P, series_c]

        # Compute final 1 Ch per series type (series has 3 ch)
        series = self.convert_image(series)

        return series

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        pid = str(self.labels.iloc[idx, 0])
        if 'DIR' in self.labels.columns and isinstance(self.labels.DIR[idx], str):
            rootDir = str(self.labels.DIR[idx])
        else:
            rootDir = self.root_dir

        if (self.series == 3):
            series = self.flow_250_all(pid, rootDir)
        elif (self.series == 0):
            series = self.flow_250_MAG(pid, rootDir)
        else:
            series = self.flow_250_other(pid, rootDir)

        label = self.load_label(idx)

        return (series, label)

class UKBBCardiacMRIMeta(UKBBCardiacMRI):
    """
    Class for MetaVGG16RNN class
    """
    def _init_Meta(self):
        MetaData = pd.read_csv("{}/MetaData.csv".format(self.root_dir))
        if self.semi:
            semi_MetaData = pd.read_csv("{}/MetaData.csv".format(self.semi_dir))
            MetaData = MetaData.append(semi_MetaData, ignore_index=True).ix[:,semi_MetaData.columns]
        self.MetaData = self.encode_MetaData(MetaData)

    def encode_MetaData(self, MetaData):
        age = np.array(MetaData.Age)
        gen = np.array(MetaData.Gender)
        ss = np.array(MetaData.SmokingStatus)
        ss[ss==-3] = 3
    
        age = (age - AGE_MEAN)/AGE_STD
        age = age.reshape(-1,1)
        gen[gen==1] = -1
        gen[gen==0] = 1
        gen = gen.reshape(-1,1)
        enc = OneHotEncoder(sparse=False)
        enc.fit(np.array([0,1,2,3,4]).reshape(-1,1))
        ss = enc.transform(ss.reshape(-1,1))
    
        encoded_MetaData = pd.DataFrame()
        encoded_MetaData['ID'] = list(MetaData.ID)
        encoded_MetaData['SmokingStatus'] = list(ss)
        encoded_MetaData['Age'] = list(age)
        encoded_MetaData['Gender'] = list(gen)
    
        return encoded_MetaData    

    def _get_meta(self, idx):
        if not hasattr(self, "MetaData"):
            self._init_Meta()
        pid = str(self.labels.iloc[idx, 0])
        meta_idx = self.MetaData.loc[self.MetaData['ID']==pid].index[0]
        return np.concatenate(self.MetaData.iloc[meta_idx,1:]).astype(float)

    def __getitem__(self, idx):
        series, label = super(UKBBCardiacMRIMeta, self).__getitem__(idx)
        meta_data = self._get_meta(idx)

        return ([series, meta_data], label)


class UKBBCardiacMRICache(UKBBCardiacMRI):
    """
    UK Biobank cardiac MRI dataset
    Load Numpy MRI sequence tensors with shape:
        num_frames X width X height

    TODO: Transformations are custom and are applied to each frame

    """
    def __init__(self, csv_data, root_dir, series=0, N=30, image_type='grey',
                 preprocess=None, augmentation=None, postprocess=None,
                 rebalance=False, threshold=0.5, seed=4321,
                 sample=False, sample_type=0, sample_split=0.5, n_samples=100,
                 pos_samples=0, neg_samples=0, frame_label=False):

        super(UKBBCardiacMRICache, self).__init__(csv_data=csv_data,
                                                  root_dir=root_dir,
                                                  series=series,
                                                  N=N,
                                                  image_type=image_type,
                                                  preprocess=preprocess,
                                                  augmentation=augmentation,
                                                  postprocess=postprocess,
                                                  rebalance=rebalance,
                                                  threshold=threshold,
                                                  seed=seed,
                                                  sample=sample,
                                                  sample_type=sample_type,
                                                  sample_split=sample_split,
                                                  n_samples=n_samples,
                                                  pos_samples=pos_samples,
                                                  neg_samples=neg_samples,
                                                  frame_label=frame_label)
        self.cache_data()

    def flow_250_MAG(self, pid):
        """
        Load the flow_250_tp_AoV_bh_ePAT@c_MAG series for the given patient.

        Params
        ------
        pid :   str
            - patient id

        Return
        ------
        np.array
            - numpy series
        """
        fpath = os.path.join(self.root_dir, seriesMap[0] + "/" + pid + ".npy")
        series = np.load(fpath).astype(np.float32)

        if self.preprocess:
            # Apply Preprocessing
            series = self.preprocess(series)

        return series

    def cache_data(self):
        self.cached_data = []
        for idx in range(len(self)):
            pid = str(self.labels.iloc[idx, 0])
            label = self.labels.iloc[idx, 1]

            # most PyTorch operations are only defined over float or doublefloat (32 vs 64bit) tensors
            if (self.series == 3):
                series = self.flow_250_all(pid)
            elif (self.series == 0):
                series = self.flow_250_MAG(pid)
            else:
                series = self.flow_250_other(pid)

            self.cached_data.append((pid, series, label))

    def __getitem__(self, idx):

        pid, series, label = self.cached_data[idx]

        if (self.series == 0):
            if self.augment:
                # Apply Agumentations
                series = self.augment(series)

            if self.postprocess:
                series = self.postprocess(series)

            # Compute final 1 Ch or 3 ch
            series = self.convert_image(series)

        return (series, label)




def stratified_sample_dataset(csv_data, seed=1234):

    labels = pd.read_csv(csv_data) if type(csv_data) is str else csv_data
    X = np.array(labels["ID"])
    Y = np.array(labels["LABEL"])

    dataframes = {}
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)

    for split1, split2 in skf.split(X, Y):
        data = np.vstack([X[split1], Y[split1]])
        dataframes["dev"] = pd.DataFrame(data.T, columns=['ID', 'LABEL'])

        data = np.vstack([X[split2], Y[split2]])
        dataframes["test"] = pd.DataFrame(data.T, columns=['ID', 'LABEL'])
        break

    return dataframes
