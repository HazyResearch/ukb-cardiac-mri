import pandas as pd
import numpy as np


__all__ = ['csv_node']

class csv_node(object):
    def __init__(self, DataFrame, seed):
        self.dataframe = DataFrame
        self.seed = int(seed)
        self.build_node()

    @classmethod
    def from_csv(klass, csv, seed=None):
        dataframe = pd.read_csv(csv)
        if seed is None:
            seed = csv.split("/")[-1].split(".")[0].split("_")[-1]
        return klass(dataframe, seed)

    def build_node(self):
        """
        Building the dict of predictions
        based on the pd.DataFrame
        """
        self.data = {}
        self.PID = []
        for i, row in self.dataframe.iterrows():
            self.PID.append(row.PID)
            self.data.update({row.PID: {"Y_TRUE": row.Y_TRUE,
                                        "Y_PROBA": row.Y_PROBA,
                                        "Y_PRED": row.Y_PRED}})

    def extract(self, key, PIDs=None):
        PIDs = self.PID if PIDs is None else PIDs
        return np.array([self.data[pid][key] for pid in PIDs])



