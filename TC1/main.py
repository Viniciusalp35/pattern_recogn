import numpy as np
import pandas as pd
import seaborn as sns
from modules.classification import KNN, centroidClassifier
from modules.acquisition import Acquisition
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, recall_score, f1_score
import time
import os

#Loading Data:
# features ... class
loader = Acquisition()

data = loader.df_raw
print("Data:\n",data)
knn = KNN()
#SPLITTING DATA:
EPOCHS = 100
MINK_ARRAY = [0.5,2/3,1,3/2,2,5/2]
TESTS_SIZE = [0.2,0.3,0.5,0.7,0.8]

def knn_mink(acc_cumulator,test_size=0.2,MINK_ARRAY = [0.5,2/3,1,3/2,2,5/2], EPOCHS = 100) -> pd.DataFrame:
    t_size = test_size*100
    train_size = (1-test_size)*100
    for mink in MINK_ARRAY:
        # ACC, predicted, t_labels
        worst = [1,[],[]]
        best = [0,[],[]]
        mink_variation = []
        for _ in range(EPOCHS):
            start_time = time.perf_counter()
            x_train, x_test, y_train, y_test = train_test_split(data.iloc[:,:-1],data.iloc[:,-1],test_size=test_size, stratify=data.iloc[:,-1])

            knn.fit(x_train,y_train)
            acc,predicted = knn.predict(x_test,np.array(y_test))
            if acc > best[0]:
                best[0] = acc; best[1] = predicted; best[2] = y_test
            elif acc < worst[0]:
                worst[0] = acc; worst[1] = predicted; worst[2] = y_test
            
            end_time = time.perf_counter()
            elapsed = end_time-start_time  
            recall = recall_score(y_test,predicted,average='macro')
            f1 = f1_score(y_test,predicted,average='macro')
            mink_variation.append(np.array([acc,recall,f1,elapsed]))
        matrix_best = confusion_matrix(best[2],best[1])
        matrix_worst = confusion_matrix(worst[2],worst[1])
        fig, axs = plt.subplots(1,2,figsize=(12,8),sharey=True,sharex=True)
        sns.heatmap(matrix_best, annot=True,
                    fmt='d',cmap='Blues',
                    yticklabels=['Move-Forward','Sl-R-Turn','Sh-R-Turn','Sl-L-Turn'],
                    xticklabels=['Move-Forward','Sl-R-Turn','Sh-R-Turn','Sl-L-Turn'],
                    ax=axs[0]
                    )
        sns.heatmap(matrix_worst, annot=True,
                    fmt='d',cmap='Blues',
                    yticklabels=['Move-Forward','Sl-R-Turn','Sh-R-Turn','Sl-L-Turn'],
                    xticklabels=['Move-Forward','Sl-R-Turn','Sh-R-Turn','Sl-L-Turn'],
                    ax=axs[1]
                    )
        plt.yticks(fontsize=9)
        plt.suptitle(f"Confusion matrix - m: {mink:.2f} with {EPOCHS} epochs")
        fig.supylabel('Real values',fontweight='bold')
        fig.supxlabel('Predicted values',fontweight='bold')
        axs[0].set_title(f"Best epoch confusion matrix - acc:{best[0]:.4f}")
        axs[1].set_title(f"Worst epoch confusion matrix - acc:{worst[0]:.4f}")
        os.makedirs(f"results/confusion_matrix_{t_size:.0f}_{train_size:.0f}", exist_ok=True)
        plt.savefig(f"results/confusion_matrix_{t_size:.0f}_{train_size:.0f}/best_worst_mink_{mink:.2f}_{EPOCHS}_epochs.png",dpi=150,bbox_inches='tight')
        plt.close()
        

        result = np.sum(np.array(mink_variation),axis=0)/EPOCHS
        acc_cumulator.append((f"{mink:.2f}",result))
    print(f"Results:\n",acc_cumulator)

    ROW = [[acc[0],*acc[1]] for acc in acc_cumulator]

    df = pd.DataFrame(data=ROW, columns=['m','accuracy','recall','f1-score','time'])
    df.to_csv(f"results/Acc_Time_results/mink_acc_time_{t_size:.0f}_{train_size:.0f}.csv",index=False)

    print("------- TOTAL TIME --------")
    print(df.iloc[:,-1].sum()*100, "s")

    return df
        
for test_size in TESTS_SIZE:
    acc_cumulator = []
    df = knn_mink(acc_cumulator,test_size=test_size)
    print(f"---------- test size: {test_size} -------------")
    print(df)
    print(df.describe())
acc_acumulator=[]
def CentroidClassifier(method='classic',acc_cumulator = acc_acumulator,test_sizes=TESTS_SIZE, EPOCHS = 100) -> pd.DataFrame:
    cc = centroidClassifier()
    for test_size in test_sizes:
        t_size = test_size*100
        train_size = (1-test_size)*100
        # ACC, predicted, t_labels
        worst = [1,[],[]]
        best = [0,[],[]]
        test_size_variation = []
        for _ in range(EPOCHS):
            start_time = time.perf_counter()
            x_train, x_test, y_train, y_test = train_test_split(data.iloc[:,:-1],data.iloc[:,-1],test_size=test_size, stratify=data.iloc[:,-1])

            cc.fit(x_train,y_train)
            acc,predicted = cc.predict(x_test,np.array(y_test))
            if acc > best[0]:
                best[0] = acc; best[1] = predicted; best[2] = y_test
            elif acc < worst[0]:
                worst[0] = acc; worst[1] = predicted; worst[2] = y_test
            
            end_time = time.perf_counter()
            elapsed = end_time-start_time  
            recall = recall_score(y_test,predicted,average='macro')
            f1 = f1_score(y_test,predicted,average='macro')
            test_size_variation.append(np.array([acc,recall,f1,elapsed]))
        matrix_best = confusion_matrix(best[2],best[1])
        matrix_worst = confusion_matrix(worst[2],worst[1])
        fig, axs = plt.subplots(1,2,figsize=(12,8),sharey=True,sharex=True)
        sns.heatmap(matrix_best, annot=True,
                    fmt='d',cmap='Blues',
                    yticklabels=['Move-Forward','Sl-R-Turn','Sh-R-Turn','Sl-L-Turn'],
                    xticklabels=['Move-Forward','Sl-R-Turn','Sh-R-Turn','Sl-L-Turn'],
                    ax=axs[0]
                    )
        sns.heatmap(matrix_worst, annot=True,
                    fmt='d',cmap='Blues',
                    yticklabels=['Move-Forward','Sl-R-Turn','Sh-R-Turn','Sl-L-Turn'],
                    xticklabels=['Move-Forward','Sl-R-Turn','Sh-R-Turn','Sl-L-Turn'],
                    ax=axs[1]
                    )
        plt.yticks(fontsize=9)
        plt.suptitle(f"Confusion matrix - {method} centroid - m: {0.5} with {EPOCHS} epochs")
        fig.supylabel('Real values',fontweight='bold')
        fig.supxlabel('Predicted values',fontweight='bold')
        axs[0].set_title(f"Best epoch confusion matrix - acc:{best[0]:.4f}")
        axs[1].set_title(f"Worst epoch confusion matrix - acc:{worst[0]:.4f}")
        os.makedirs(f"results/confusion_matrix_{t_size:.0f}_{train_size:.0f}", exist_ok=True)
        plt.savefig(f"results/confusion_matrix_{t_size:.0f}_{train_size:.0f}/best_worst_centroid_{method}_{0.5}_{EPOCHS}_epochs.png",dpi=150,bbox_inches='tight')
        plt.close()
        

        result = np.sum(np.array(test_size_variation),axis=0)/EPOCHS
        acc_cumulator.append((f"{t_size}%",result))
    print(f"Results:\n",acc_cumulator)

    ROW = [[acc[0],*acc[1]] for acc in acc_cumulator]

    df = pd.DataFrame(data=ROW, columns=['test_size','accuracy','recall','f1-score','time'])
    df.to_csv(f"results/Acc_Time_results/centroid_{method}.csv",index=False)

    print("------- TOTAL TIME --------")
    print(df.iloc[:,-1].sum()*100, "s")

    return df

CentroidClassifier()
acc_acumulator2=[]
CentroidClassifier(acc_cumulator=acc_acumulator2,method='outlier robust')
CentroidClassifier('mahalo')

