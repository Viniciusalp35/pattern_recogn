import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr


class Acquisition:
    DATA_PATH = "./data/sensor_readings_4.data"
    COL_NAMES = ['SD_front', 'SD_left','SD_right','SD_back', 'Class']
    def __init__(self, path:str = DATA_PATH, names= COL_NAMES):
        self.df_raw = pd.read_csv(path,header=None, names=names)
    
    def describe_dataset(self,print_on_terminal=True):
        metrics = ['count','mean','var','skew','kurt']
        statistical_description = self.df_raw.groupby('Class').agg(metrics)
        if print_on_terminal:
            print("----------- DATA DESCRIPTION -------------")
            print("NUMBER OF CLASSES:",self.df_raw['Class'].unique())
            print("NUMBER OF INSTANCES PER CLASS\n", self.df_raw[['Class']].value_counts())
            print("CLASS STATISTIC DESCRIPTION:\n", statistical_description)
        with open("results/data_description.txt","w+") as f:
            f.write("--------------- DATA DESCRIPTION ------------------\n\n\n")
            f.write(f"NUMBER OF INSTANCES PER CLASS\n {self.df_raw[['Class']].value_counts()}\n\n\n")
            f.write(f"NUMBER OF INSTANCES PER CLASS\n {self.df_raw[['Class']].value_counts()}\n\n\n")
            f.write(f"CLASS STATISTIC DESCRIPTION:\n {statistical_description}")
        
    def plotter(self,df_raw:pd.DataFrame|None = None, plot: bool = False):
        if not df_raw:
            df_raw = self.df_raw
        self._plot_hist()
        plt.savefig("results/feature_histograms",dpi=150)
        plt.close()
        if plot:
            plt.show()

        self._pairplot_matrix()
        plt.savefig("results/pairplot_features",dpi=150)
        plt.close()
        if plot:
            plt.show()

        self._correlation_matrix()
        plt.savefig("results/correlation_matrix",dpi=150)
        plt.close()
        if plot:
            plt.show()

        for c in df_raw['Class'].unique():
            self._plot_hist(df_raw[df_raw['Class'] == c],f"HISTOGRAM DISTRIBUTION BY FEATURE - CLASS: {c}")
            plt.savefig(f"results/hist_by_class/hist_{c}")
            if plot:
                plt.show()

    def _plot_hist(self,df_raw:pd.DataFrame|None = None, title: str = "HISTOGRAM DISTRIBUTION BY FEATURE"):
        if not isinstance(df_raw,pd.DataFrame):
            df_raw = self.df_raw
        fig, axs = plt.subplots(2,2,figsize=(12,8))
        fig.suptitle(title)
        axs_flatten = axs.flatten()
        for idx,feature in enumerate(df_raw.columns[:-1]):
            sns.histplot(df_raw, x=feature, ax=axs_flatten[idx])
        
    def _pairplot_matrix(self ,df_raw: pd.DataFrame|None = None):
        if not df_raw:
            df_raw = self.df_raw
        sns.pairplot(df_raw,
                     diag_kind='hist',
                     kind='scatter',
                     corner=False,
                     height=2.2,
                     aspect=1.1,
                     hue='Class'
                     )
    
    def _correlation_matrix(self,df_raw:pd.DataFrame|None = None):
        if not df_raw:
            df_raw = self.df_raw
        df_corr = df_raw.iloc[:,:-1].corr()
        sns.heatmap(df_corr, annot=True,fmt=".2f")


    def routine(self):
        self.describe_dataset()
        self.plotter()
if __name__ == "__main__":
    Ac_obj = Acquisition()
    print(Ac_obj.df_raw, "\n\n\n")
    Ac_obj.routine()

    

