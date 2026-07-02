import pandas as pd

class Loader():
    DATA_PATH = "./data/sensor_readings_4.data"
    COL_NAMES = ['SD_front', 'SD_left','SD_right','SD_back', 'Class']
    def __init__(self, path:str = DATA_PATH, names= COL_NAMES):
        self.df_raw = pd.read_csv(path,header=None, names=names)