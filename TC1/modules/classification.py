import numpy as np
import pandas as pd

class Classifier:
    @classmethod
    def Minkowski_distance(cls,x_train:np.ndarray,x_test:np.ndarray, m):
        """
        Calculates the Minkowski distance by array manipulation using numpy´s array

        :input:
        x -> numpy array with coordenates\n
        y -> numpy array with coordenates\n
        m -> Minkowski order 
        """
        diff = np.abs(x_test[:,np.newaxis,:]-x_train[np.newaxis,:,:])
        return np.sum(diff**m,axis=2) ** (1/m)
    
class KNN:
    def __init__(self,k:int=3,m:int=1):
        self.k = k
        self.m = m
        self.x_train = None
        self.y_train = None

    def fit(self,x_train:pd.DataFrame,y_train:pd.DataFrame):
        self.x_train = np.array(x_train) #VALORES SD_FRONT ... SD_BACK
        self.y_train = np.array(y_train) #CLASSIFICAÇÃO DAQUELES PONTOS

    def predict(self,x:int,y:int):
        min_array = Classifier.Minkowski_distance(x,y)
        