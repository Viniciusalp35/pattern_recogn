import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.model_selection import train_test_split


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
        get_method = dict_frequent.get
        for l in labels:
            dict_frequent[l] = get_method(l,0) + 1
        
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
            data_row = truth_dataframe.iloc[0,:]
            acc = (np.array(data_row)/np.sum(np.array(data_row)))[1]
            #print(f"Accuracy: {acc:.2f}")
            return acc,predicted
        
class centroidClassifier:
    def __init__(self,method:str = 'classic'):
        self.classes = []
        self.centroids = []
        self.method = method
    
    def fit(self,x_train:np.ndarray,y_train:np.ndarray):
        """
        Separa as classes sem balanceamento de classes.
        """
        self.classes = np.unique(y_train).tolist()
        self.df_x = pd.DataFrame(x_train)
        self.df_x['Class'] = np.array(y_train)
        if self.method == 'classic':
            self.centroids = self.df_x.groupby('Class').mean()
        else:
            self.centroids = self.df_x.groupby('Class').median()
        
        self.centroids = self.centroids.loc[self.classes].values
        #print(self.centroids)
    
    def _min_distance(self,x_test,y_test):
        distances = Classifier.Minkowski_distance(np.asarray(self.centroids),np.asarray(x_test),0.5)
        min_distances = np.min(distances,axis=1)
        closest_class_index = np.argmin(distances,axis=1)
        predicted = np.array(self.classes)[closest_class_index]
        if isinstance(y_test,np.ndarray) and y_test.shape[0] > 0:
            y_test = np.array(y_test)
            truth_dataframe = KNN._comparator(predicted,y_test)
            data_row = truth_dataframe.iloc[0,:]
            acc = (np.array(data_row)/np.sum(np.array(data_row)))[1]
            #print(f"Accuracy: {acc:.2f}")
            return acc,predicted
    def predict(self,x_test:np.ndarray,y_test:np.ndarray):
        return self._min_distance(x_test,y_test)
        
            
            
            


        


if __name__ == "__main__":
   frequentes = KNN._most_frequent(['banana','maça','arroz','banana'])
   np.random.seed(42)
   dados = {
        'feature_1': np.random.rand(10),
        'feature_2': np.random.rand(10),
        'Class': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C', 'C']
    }
   df = pd.DataFrame(dados)
   cc = centroidClassifier()
   cc.fit(pd.DataFrame(df))
   cc.predict(df)
   print(frequentes)
        