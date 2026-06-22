import numpy as np
import pandas as pd
from scipy.stats import mode


class Classifier:
    @classmethod
    def Minkowski_distance(cls,x_train:np.ndarray,x_test:np.ndarray, m=0.5)->np.ndarray:
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
    @staticmethod
    def _most_frequent(array: np.array,axis:int=0)-> np.array:
        dict_frequent = {}
        array = np.array(array)
        labels=np.take(array,axis=axis,indices=np.arange(array.shape[axis]))
        labels = labels.flatten()
        for l in labels:
            if l in dict_frequent:
                dict_frequent[l] +=1
            else:
                dict_frequent[l] = 1
        
        max_key = max(dict_frequent,key=dict_frequent.get)
        return max_key
    
    @staticmethod
    def _comparator(predicted:np.array,y_test:np.array)->pd.DataFrame:
        bool_mask = predicted == y_test
        
        Values, count = np.unique_counts(bool_mask)
        ROW = count.reshape(1,-1)
        COLUMN = Values
        df=pd.DataFrame(data=ROW, columns=COLUMN)
        #print(df)
        return df

    def fit(self,x_train:pd.DataFrame,y_train:pd.DataFrame):
        self.x_train = np.array(x_train) #VALORES SD_FRONT ... SD_BACK
        self.y_train = np.array(y_train) #CLASSIFICAÇÃO DAQUELES PONTOS

    def predict(self,x_test, y_test = [],m = 0.5):
        x_test = np.array(x_test)
        if x_test.ndim == 1:
            x_test = x_test.reshape(1,-1)
        diff_array = Classifier.Minkowski_distance(self.x_train,x_test,m)
        sorted = np.argsort(diff_array,axis=1)[:,:self.k]
        k_nearest_neighbours = self.y_train[sorted]
        predicted = [self._most_frequent(row) for row in k_nearest_neighbours]
        predicted=np.array(predicted)
        #print("K winner:", predicted)
        if isinstance(y_test,np.ndarray) and y_test.shape[0] > 0:
            y_test = np.array(y_test)
            #print(f"--------- CLASSIFIER STATISTICS KNN FOR K = {self.k} and M = {m} -----------")
            truth_dataframe = self._comparator(predicted,y_test)
            acc = (np.array(truth_dataframe.iloc[0,:])/np.sum(np.array(truth_dataframe.iloc[0,:])))[1]
            #print(f"Accuracy: {acc:.2f}")
            return acc
            
            
            


        


if __name__ == "__main__":
   frequentes = KNN._most_frequent(['banana','maça','arroz','banana'])

   print(frequentes)
        