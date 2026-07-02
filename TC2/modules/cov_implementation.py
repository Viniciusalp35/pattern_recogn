import numpy as np
import pandas as pd
from modules.Loader import Loader
import time

def ref_matrix(data):
    t_start = time.perf_counter()
    m,n = data.shape
    C = np.cov(data,rowvar=False,ddof=0)
    #print("--------- REF MATRIX ---------")
    #print(f"Numero de features: {n}, shape da matriz: {C.shape}, matriz:\n {C}\nvariances:{np.var(data,axis=0)} ")
    t_end = time.perf_counter()
    dt = t_end-t_start
    return C, dt


#OS DADOS ESTAO NAS LINHAS E NAO NAS COLUNAS, COM RELAÇÃO AO SLIDE A IDEIA É QUE SE FEZ UM ()^T EM TODAS AS OPERAÇÕES, RESULTANDO NAS FUNÇÕES A SEGUIR:
#OBS: ERA POSSIVEL TER FEITO A TRANSPOSTA APENAS DE 'data' CONTUDO ACHO MAIS VISUAL TRATAR OS DADOS COM OS 'records' NAS LINHAS

def cov_1(data:np.ndarray):
    # m records, n atributos
    t_start = time.perf_counter()
    m,n = data.shape
    resulting_sum = np.zeros((n,n))
    media = np.mean(data,axis=0)
    for i in range(0,m): #percorre as linhas
        aux = data[i,:] - media 
        aux = aux.flatten()
        resulting_sum += np.outer(aux,aux) #Multiplicação de matriz
    
    C=resulting_sum/m
    C = (C + C.T)/2 #SIMETRIA FLOAT
    t_end = time.perf_counter()
    dt = t_end - t_start
    return C, dt

def cov_2(data:np.ndarray):
    t_start = time.perf_counter()
    m,n = data.shape
    media = np.mean(data,axis=0)
    aux = data - media #m,n 
    C = (aux.T@aux)/m
    C = (C+C.T)/2 #SIMETRIA FLOAT
    t_end = time.perf_counter()
    dt = t_end-t_start
    return C, dt

def cov_3(data:np.ndarray):
    t_start = time.perf_counter()
    m,n = data.shape
    media = np.mean(data,axis=0)
    R = np.zeros((n,n))
    for i in range(m): # percorre as linhas
        R += np.outer(data[i,:],data[i,:])
    C = R/m - np.outer(media,media)
    C = (C+C.T)/2 #SIMETRIA FLOAT
    t_end = time.perf_counter()
    dt = t_end-t_start
    return C, dt

def cov_4(data:np.ndarray):
    t_start = time.perf_counter()
    m,n = data.shape
    media = np.mean(data,axis=0)
    R = data.T@data
    C = R/m - np.outer(media,media)
    C = (C+C.T)/2 #SIMETRIA FLOAT
    t_end = time.perf_counter()
    dt = t_end-t_start
    return C, dt





if __name__ == "__main__":
    loader = Loader()
    data = np.asarray(loader.df_raw.iloc[:,:-1])
    m,n = data.shape
    C, perf = cov_1(data)
    print("--------- COV_1 MATRIX ---------")
    print(f"Numero de features: {n}, shape da matriz: {C.shape}, matriz:\n {C}\nvariances:{np.var(data,axis=0)}\ntempo_exec = {perf} ")
    C, perf = cov_2(data)
    print("--------- COV_2 MATRIX ---------")
    print(f"Numero de features: {n}, shape da matriz: {C.shape}, matriz:\n {C}\nvariances:{np.var(data,axis=0)}\ntempo_exec = {perf}")
    C, perf = cov_3(data)
    print("--------- COV_3 MATRIX ---------")
    print(f"Numero de features: {n}, shape da matriz: {C.shape}, matriz:\n {C}\nvariances:{np.var(data,axis=0)}\ntempo_exec = {perf}")
    C, perf = cov_4(data)
    print("--------- COV_4 MATRIX ---------")
    print(f"Numero de features: {n}, shape da matriz: {C.shape}, matriz:\n {C}\nvariances:{np.var(data,axis=0)}\ntempo_exec = {perf}")
    C, perf = ref_matrix(data)
    print("--------- REF MATRIX ---------")
    print(f"Numero de features: {n}, shape da matriz: {C.shape}, matriz:\n {C}\nvariances:{np.var(data,axis=0)}\ntempo_exec = {perf} ")



