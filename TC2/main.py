import numpy as np
import matplotlib.pyplot as plt
from modules.Loader import Loader
from modules.cov_implementation import *
import seaborn as sns

"""
CALCULO DAS COMPARAÇÕES. DATA SEM LABELs
"""
covs = [cov_1,cov_2,cov_3,cov_4]


def comparator(data:np.ndarray):
    # MATRIZ REFERENCIA
    C_ref = ref_matrix(data)[0]
    cov_matrix = [cov(data)[0] for cov in covs]
    print(cov_matrix)
    norm_result = []
    for i in range(len(cov_matrix)):
        print(f"DIFERENÇA ENTRE COV_{i+1} E REFERENCIA:")
        norm = np.linalg.norm(cov_matrix[i] - C_ref)
        print("norma:", norm)
        norm_result.append((f'cov_{i+1}',norm))
    print(norm_result)
    return norm_result

def time_comparator(data:np.ndarray):
    m,n = data.shape
    print(data.shape)
    time_array = []
    for i,cov in enumerate(covs):
        for _ in range(100):
            time_array.append((f"cov_{i+1}",cov(data)[1]))
    df = pd.DataFrame(data = time_array, columns=["cov_function",'time_(ms)'])
    print(f"MEDIA PARA AS EXECUÇÕES - {n} readings \n",
          df.groupby('cov_function').mean()
          )
    df.groupby('cov_function').mean().to_csv(f"./results/time_comparison_{n}_readings.csv")
    df_fast = df[df['cov_function'].isin(['cov_2','cov_4'])]
    df_slow = df[df['cov_function'].isin(['cov_1','cov_3'])]
    fig, axs = plt.subplots(2,2,figsize=(12,8))
    fig.suptitle(f"Comparison plots for {n} readings")
    #PLOT RAPIDO
    sns.violinplot(df_fast,y='cov_function',x='time_(ms)',ax=axs[0,0])
    sns.histplot(df_fast,hue='cov_function',x='time_(ms)',ax=axs[0,1])
    axs[0,0].set_title('Violin Plot')
    axs[0,1].set_title("Histogram Plot")
    #PLOT LENTO
    sns.violinplot(df_slow,y='cov_function',x='time_(ms)',ax=axs[1,0])
    sns.histplot(df_slow,hue="cov_function",x='time_(ms)',ax=axs[1,1])
    plt.savefig(f"./results/time_comparison_{n}_readings.png",dpi=300)
    




if __name__ == "__main__":
    loader_4 = Loader()
    data_4 = loader_4.df
    loader_24 = Loader("./data/sensor_readings_24.data",[f"US{i}"for i in range(24)]+["Class"])
    data_24 = loader_24.df
    #QUESTAO 01
    ROW = [];COLUMNS = ['cov usado','norma da diferença']
    ROW += comparator(data_4); ROW += comparator(data_24)
    df = pd.DataFrame(data=ROW, columns=COLUMNS, index=['4_readings' if i < len(ROW)/2 else '24_readings' for i in range(len(ROW))]); df.to_csv("results/Questao01.csv")
    #QUESTAO 02
    time_comparator(data_4)
    time_comparator(data_24)