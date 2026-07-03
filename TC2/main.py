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


def _wellnes_check(c_mat,n):
    rank = np.linalg.matrix_rank(c_mat)
    cond = np.linalg.cond(c_mat)
    rcond= 1/cond if cond != 0 else 0.0
    wellness = rank == n and rcond > 1e-6
    return rank,rcond,wellness

def questao03(data:np.ndarray,loader_obj:Loader):
    m,n = data.shape
    fastest_method = covs[-1]
    C_global = fastest_method(data)[0]
    rank_g,rcond_g,wellness_g = _wellnes_check(C_global,n)
    c_by_class = []
    score_by_class = [("GLOBAL",rank_g,rcond_g,wellness_g)]
    data_by_class , classes = loader_obj.cov_by_class()
    
    for df,class_name in zip(data_by_class,classes):
        cov = fastest_method(df)[0]
        c_by_class.append((cov,class_name))
        rank,rcond,wellness = _wellnes_check(cov,n)
        score_by_class.append((class_name,rank,rcond,wellness))
    df = pd.DataFrame(data=score_by_class, columns=["Class",'POSTO','RCOND','INVERTIBILIDADE'])
    df.to_csv(f"./results/questao03/invertibilidade_{n}_readings.csv")
    return c_by_class

def questao04(cov_list: list[(np.ndarray,str)]):
    result = []
    for cov,class_name in cov_list:
        try:
            inv_c = np.linalg.inv(cov)
            with open(f"./results/questao04/{cov.shape[0]}_records_inversas.txt",'a+') as f:
                f.write(f"--------- {class_name} INVERSE ---------\n")
                f.write(f"{np.array2string(inv_c)}\n\n")
            result.append((f"{class_name}",'true'))
        except Exception as e:
            print("NAO DEU PARA INVERTER: ",e)
            result.append((f"{class_name}",'false'))
    
    df = pd.DataFrame(data=result, columns=['CLASSE DA INVERSA','INVERTEU?'])
    df.to_csv("./results/questao04/relation_class_invertible.csv")

    




if __name__ == "__main__":
    loader_4 = Loader()
    data_4 = loader_4.df
    loader_24 = Loader("./data/sensor_readings_24.data",[f"US{i}"for i in range(24)]+["Class"])
    data_24 = loader_24.df
    print("\n\n\n---------- Questão 01 ----------")
    #QUESTAO 01
    ROW = [];COLUMNS = ['cov usado','norma da diferença']
    ROW += comparator(data_4); ROW += comparator(data_24)
    df = pd.DataFrame(data=ROW, columns=COLUMNS, index=['4_readings' if i < len(ROW)/2 else '24_readings' for i in range(len(ROW))]); df.to_csv("results/Questao01.csv")
    print("\n\n\n---------- Questão 02 ----------")
    #QUESTAO 02
    time_comparator(data_4)
    time_comparator(data_24)
    print("\n\n\n---------- Questão 03 ----------")
    #QUESTAO 03
    c_3_4 = questao03(data_4,loader_4)
    c_3_24 = questao03(data_24,loader_24)
    print("\n\n\n---------- Questão 04 ----------")
    #QUESTAO 04:
    questao04(c_3_4)
    questao04(c_3_24)
