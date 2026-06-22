import numpy as np
import pandas as pd
from modules.classification import KNN
from modules.acquisition import Acquisition
from sklearn.model_selection import train_test_split
import time
import os

#Loading Data:
# features ... class
loader = Acquisition()

data = loader.df_raw
print("Data:\n",data)
knn = KNN()
#SPLITTING DATA:
acc_cumulator = []
EPOCHS = 100
MINK_ARRAY = [0.5,2/3,1,3/2,2,5/2]

def knn_mink(test_size=0.2,MINK_ARRAY = [0.5,2/3,1,3/2,2,5/2], EPOCHS = 100) -> pd.DataFrame:
    for mink in MINK_ARRAY:
        mink_variation = []
        for i in range(EPOCHS):
            start_time = time.perf_counter()
            x_train, x_test, y_train, y_test = train_test_split(data.iloc[:,:-1],data.iloc[:,-1],test_size=test_size, stratify=data.iloc[:,-1])

            knn.fit(x_train,y_train)
            acc = knn.predict(x_test,np.array(y_test))
            end_time = time.perf_counter()
            elapsed = end_time-start_time  
            mink_variation.append(np.array([acc,elapsed]))

        result = np.sum(np.array(mink_variation),axis=0)/EPOCHS
        acc_cumulator.append((f"{mink:.2f}",result))
    print(f"Results:\n",acc_cumulator)

    ROW = [[acc[0],acc[1][0],acc[1][1]] for acc in acc_cumulator]

    df = pd.DataFrame(data=ROW, columns=['m','accuracy','time'])
    df.to_csv("results/mink_acc_time.csv",index=False)

    return df
        
df = knn_mink()

try:
    df_readed = pd.read_csv("results/mink_acc_time.csv")
except:
    df_readed = df
finally:
    df = df_readed

print(df)

print(df.describe())
