import numpy as np
import pandas as pd
import seaborn as sns
from modules.classification import KNN, centroidClassifier
from modules.acquisition import Acquisition
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, recall_score, f1_score, precision_score
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
    """
    Calcula as estatisticas para as rotinas do knn, alterando o valor de mink e permitindo a mudança do test_size.
    Plota as matrizes de confusao e guarda todas as estatisticas e imagens geradas em results
    """
    t_size = test_size*100
    train_size = (1-test_size)*100
    description_acumulator = []
    labels = data.iloc[:,-1].unique()
    for mink in MINK_ARRAY:
        # ACC, predicted, t_labels
        worst = [1,[],[]]
        best = [0,[],[]]
        mink_variation = []
        recall_by_class = []
        f1_by_class = []
        precision_by_class = []
        for i in range(EPOCHS):
            start_time = time.perf_counter()
            x_train, x_test, y_train, y_test = train_test_split(data.iloc[:,:-1],data.iloc[:,-1],test_size=test_size, stratify=data.iloc[:,-1],random_state=i)

            knn.fit(x_train,y_train)
            acc,predicted = knn.predict(x_test,np.array(y_test),mink)
            if acc > best[0]:
                best[0] = acc; best[1] = predicted; best[2] = y_test
            elif acc < worst[0]:
                worst[0] = acc; worst[1] = predicted; worst[2] = y_test
            
            end_time = time.perf_counter()
            elapsed = end_time-start_time  

            recall = recall_score(y_test,predicted,average='macro')
            recall_by_class.append(recall_score(y_test,predicted, average=None, labels=labels))

            f1_by_class.append(f1_score(y_test,predicted,average=None,labels=labels))
            f1 = f1_score(y_test,predicted,average='macro')

            precision_by_class.append(precision_score(y_test,predicted, average=None, labels=labels))

            mink_variation.append(np.array([acc,recall,f1,elapsed]))
        matrix_best = confusion_matrix(best[2],best[1])
        matrix_worst = confusion_matrix(worst[2],worst[1])
        #GERA AS MATRIZES DE CONFUSAO PARA OS MELHORES E PIORES RESULTADOS DAS EPOCAS
        fig, axs = plt.subplots(1,2,figsize=(12,8),sharey=True,sharex=True)
        sns.heatmap(matrix_best, annot=True,
                    fmt='d',cmap='Blues',
                    yticklabels=[l[:15] for l in labels],
                    xticklabels=[l[:15] for l in labels],
                    ax=axs[0]
                    )
        sns.heatmap(matrix_worst, annot=True,
                    fmt='d',cmap='Blues',
                    yticklabels=[l[:15] for l in labels],
                    xticklabels=[l[:15] for l in labels],
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
        #Calculo das estatisticas captadas:
        mink_arr = np.array(mink_variation)
        
        #Extrai apenas a coluna de acc (índice 0)
        acc_arr = mink_arr[:, 0]
        
        #Estatisticas de acurácia
        mean_acc = np.mean(acc_arr)
        std_acc = np.std(acc_arr)  # Desvio-padrão
        max_acc = np.max(acc_arr)  # Ou best[0]
        min_acc = np.min(acc_arr)  # Ou worst[0]
        median_acc = np.median(acc_arr) # Mediana
        
        #Media de recall f1 e tempo
        mean_rec = np.mean(mink_arr[:, 1])
        mean_f1 = np.mean(mink_arr[:, 2])
        mean_time = np.mean(mink_arr[:, 3])

        # Agrupa os resultados
        statistics = [mean_acc, std_acc, max_acc, min_acc, median_acc, mean_rec, mean_f1, mean_time]
        acc_cumulator.append((f"{mink:.2f}",statistics))

        #GERAÇÃO DAS DESCRIÇÕES POR CLASSE E TEST_SIZE:
        mean_recall_by_class = np.mean(np.array(recall_by_class),axis=0)
        mean_f1_by_class = np.mean(np.array(f1_by_class),axis=0)
        mean_precision_by_class = np.mean(np.array(precision_by_class),axis=0)
        row_classes = [
                    mink,
                    *mean_precision_by_class,
                    *mean_recall_by_class,
                    *mean_f1_by_class,
                    mean_time
        ]
                
        description_acumulator.append(row_classes)
    print(f"Results:\n",acc_cumulator)

    ROW = [[acc[0],*acc[1]] for acc in acc_cumulator]

    df = pd.DataFrame(data=ROW, columns=['m', 'mean_acc', 'std_acc', 'max_acc', 'min_acc', 'median_acc', 'mean_recall', 'mean_f1-score', 'mean_time'])
    df.to_csv(f"results/Acc_Time_results/mink_acc_time_{t_size:.0f}_{train_size:.0f}.csv",index=False)

    #CRIAÇÃO DO DATAFRAME ESTATISTICA POR CLASSE:
    column_csv = ['mink factor']
    for cls in labels: column_csv.append(f'{cls}_Precision')
    for cls in labels: column_csv.append(f'{cls}_Recall')
    for cls in labels: column_csv.append(f'{cls}_F1')
    column_csv.append('mean_time (s)')

    df_classes = pd.DataFrame(data=description_acumulator, columns=column_csv)
    
    df_classes.to_csv(f"results/confusion_matrix_{t_size:.0f}_{train_size:.0f}/KNN_mink_metrics_by_class.csv", index=False)

    print("------- TOTAL TIME --------")
    print(df.iloc[:,-1].sum()*100, "s")

    return df
#ALTERA O TEST SIZE
for test_size in TESTS_SIZE:
    acc_cumulator = []
    df = knn_mink(acc_cumulator,test_size=test_size)
    print(f"---------- test size: {test_size} -------------")
    print(df)
    print(df.describe())
acc_acumulator=[]
def CentroidClassifier(method='classic',acc_cumulator = [],test_sizes=TESTS_SIZE, EPOCHS = 100,m=2) -> pd.DataFrame:
    """
    Calcula as estatisticas para as rotinas dos centroids, alterando o test_size e permitindo a mudança do metodo utilizado no centroid.
    Plota as matrizes de confusao e guarda todas as estatisticas e imagens geradas em results
    """
    description_acumulator = []
    cc = centroidClassifier(m,method=method)
    labels = data.iloc[:,-1].unique()
    for test_size in test_sizes:
        t_size = test_size*100
        train_size = (1-test_size)*100
        # ACC, predicted, t_labels
        worst = [1,[],[]]
        best = [0,[],[]]
        test_size_variation = []
        recall_by_class = []
        f1_by_class = []
        precision_by_class = []
        for i in range(EPOCHS):
            start_time = time.perf_counter()
            x_train, x_test, y_train, y_test = train_test_split(data.iloc[:,:-1],data.iloc[:,-1],test_size=test_size, stratify=data.iloc[:,-1],random_state=i)

            cc.fit(x_train,y_train)
            acc,predicted = cc.predict(x_test,np.array(y_test))
            if acc > best[0]:
                best[0] = acc; best[1] = predicted; best[2] = y_test
            elif acc < worst[0]:
                worst[0] = acc; worst[1] = predicted; worst[2] = y_test
            
            end_time = time.perf_counter()
            elapsed = end_time-start_time  
            #Calculo de recall por classe e global
            recall_by_class.append(recall_score(y_test,predicted, average=None, labels=labels))
            recall = recall_score(y_test,predicted,average='macro')
            #Calculo de f1-score por classe e global
            f1_by_class.append(f1_score(y_test,predicted,average=None,labels=labels))
            f1 = f1_score(y_test,predicted,average='macro')

            test_size_variation.append(np.array([acc,recall,f1,elapsed]))
            #Calculo de precisão por classe
            precision_by_class.append(precision_score(y_test,predicted, average=None, labels=labels))

        matrix_best = confusion_matrix(best[2],best[1])
        matrix_worst = confusion_matrix(worst[2],worst[1])
        #GERA AS MATRIZES DE CONFUSAO PARA OS MELHORES E PIORES RESULTADOS DAS EPOCAS
        fig, axs = plt.subplots(1,2,figsize=(12,8),sharey=True,sharex=True)
        sns.heatmap(matrix_best, annot=True,
                    fmt='d',cmap='Blues',
                    yticklabels=[l[:15] for l in labels],
                    xticklabels=[l[:15] for l in labels],
                    ax=axs[0]
                    )
        sns.heatmap(matrix_worst, annot=True,
                    fmt='d',cmap='Blues',
                    yticklabels=[l[:15] for l in labels],
                    xticklabels=[l[:15] for l in labels],
                    ax=axs[1]
                    )
        plt.yticks(fontsize=9)
        plt.suptitle(f"Confusion matrix - {method} centroid - {'m: ' + f'{m}' if method != 'correlation' else 'cosine_similarity'} with {EPOCHS} epochs")
        fig.supylabel('Real values',fontweight='bold')
        fig.supxlabel('Predicted values',fontweight='bold')
        axs[0].set_title(f"Best epoch confusion matrix - acc:{best[0]:.4f}")
        axs[1].set_title(f"Worst epoch confusion matrix - acc:{worst[0]:.4f}")
        os.makedirs(f"results/confusion_matrix_{t_size:.0f}_{train_size:.0f}", exist_ok=True)
        plt.savefig(f"results/confusion_matrix_{t_size:.0f}_{train_size:.0f}/best_worst_centroid_{method}_{m if method != 'correlation' else 'cosine_similarity'}_{EPOCHS}_epochs.png",dpi=150,bbox_inches='tight')
        plt.close()

         #Calculo das estatisticas captadas:
        mink_arr = np.array(test_size_variation)
        
        #Extrai apenas a coluna de acc (índice 0)
        acc_arr = mink_arr[:, 0]
        
        #Estatisticas de acurácia
        mean_acc = np.mean(acc_arr)
        std_acc = np.std(acc_arr)  # Desvio-padrão
        max_acc = np.max(acc_arr)  # Ou best[0]
        min_acc = np.min(acc_arr)  # Ou worst[0]
        median_acc = np.median(acc_arr) # Mediana
        
        #Media de recall f1 e tempo
        mean_rec = np.mean(mink_arr[:, 1])
        mean_f1 = np.mean(mink_arr[:, 2])
        mean_time = np.mean(mink_arr[:, 3])

        # Agrupa os resultados
        statistics = [mean_acc, std_acc, max_acc, min_acc, median_acc, mean_rec, mean_f1, mean_time]
        acc_cumulator.append((f"{t_size}",statistics))
        #GERAÇÃO DAS DESCRIÇÕES POR CLASSE E TEST_SIZE:
        mean_recall_by_class = np.mean(np.array(recall_by_class),axis=0)
        mean_f1_by_class = np.mean(np.array(f1_by_class),axis=0)
        mean_precision_by_class = np.mean(np.array(precision_by_class),axis=0)
        row_classes = [
                    t_size,
                    *mean_precision_by_class,
                    *mean_recall_by_class,
                    *mean_f1_by_class,
                    mean_time
        ]
                
        description_acumulator.append(row_classes)
   
    column_csv = ['test_size %']
    for cls in labels: column_csv.append(f'{cls}_Precision')
    for cls in labels: column_csv.append(f'{cls}_Recall')
    for cls in labels: column_csv.append(f'{cls}_F1')
    column_csv.append('mean_time (s)')

    df_classes = pd.DataFrame(data=description_acumulator, columns=column_csv)
    
    df_classes.to_csv(f"results/Acc_Time_results/centroid_{method}_{f'{m}' if method != 'correlation' else 'cosine_sim'}_metrics_by_class.csv", index=False)

    print(f"Results:\n",acc_cumulator)

    ROW = [[acc[0],*acc[1]] for acc in acc_cumulator]

    df = pd.DataFrame(data=ROW, columns=['test_size','mean_acc', 'std_acc', 'max_acc', 'min_acc', 'median_acc', 'mean_recall', 'mean_f1-score', 'mean_time'])
    df.to_csv(f"results/Acc_Time_results/centroid_{method}_{m if method != 'correlation' else 'cosine_sim'}.csv",index=False)

    print("------- TOTAL TIME --------")
    print(df.iloc[:,-1].sum()*100, "s")

    return df

#CENTROIDE CLASSICO COM M = 2
CentroidClassifier()
acc_acumulator2=[]
#CENTROIDE ROBUSTO A OUTLIER COM M=2
CentroidClassifier(acc_cumulator=acc_acumulator2,method='outlier robust')
#CENTROIDE COM DISTANCIA DE MAHALA - DESCONSIDERAR
#acc_acumulator3=[]
#CentroidClassifier(acc_cumulator=acc_acumulator3,method='mahala')
acc_acumulator4=[]
CentroidClassifier(acc_cumulator=acc_acumulator4,method='correlation')

#CRIANDO PARA M = 0.5
acc_acumulator = []
#CENTROIDE CLASSICO COM M=0.5
CentroidClassifier(acc_cumulator=acc_acumulator,m=0.5)
acc_acumulator2=[]
#CENTROIDE ROBUSTO A OUTLIER COM M=0.5
CentroidClassifier(acc_cumulator=acc_acumulator2,method='outlier robust',m=0.5)

