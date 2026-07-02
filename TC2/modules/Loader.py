import pandas as pd
import numpy as np

class Loader():
    DATA_PATH = "./data/sensor_readings_4.data"
    COL_NAMES = ['SD_front', 'SD_left','SD_right','SD_back', 'Class']
    def __init__(self, path:str = DATA_PATH, names= COL_NAMES):
        self.df_raw = pd.read_csv(path,header=None, names=names)
        self.df = np.asarray(self.df_raw.iloc[:,:-1])

    def cov_by_class(self):
        data_array = []
        classes = self.df_raw['Class'].unique()
        for c in classes:
            data_array.append(self.df_raw[self.df_raw['Class']==c])
        print(data_array)
        return data_array


if __name__ == "__main__":
    load = Loader()
    load.cov_by_class()