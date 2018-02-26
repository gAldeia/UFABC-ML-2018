import pandas as pd
import numpy as np

from scipy import stats


def minkowski(p, x, y):
    '''
    Metrica de distancia entre dois vetores n-dimensionais ()

    Parametros:
        p (int): ordem da equacao. quando p=2, equivale a distancia euclidiana
        x (numpy array): primeiro vetor n-dimensional 
        y (numpy array): segundo vetor n-dimensional
    Retorno:
        (float): medida da distancia de minkowski
    '''
    
    difP = np.power(np.abs(x-y), p)
    soma = np.sum(difP)

    return np.power(soma, 1.0/p)


def knn(xi, X, Y, k):
    '''
    Encontra um rotulo adequado para uma amostra de acordo com
    um conjunto de dados passado como referencial

    Parametros:
        x1 (numpy array): amostra a ser rotulado
        X (numpy array): conjunto de dados de exemplo
        Y (numpy array): conjunto de rotulos para os exemplos
        k (int): quantidade de vizinhos 
    Retorno:
        (numpy array): melhor rotulo para a amostra
    '''

    neighbors = [minkowski(2, xi, x) for x in X]

    idx = np.argsort(neighbors)[:k]

    return stats.mode(Y[idx])[0]


def main():
    #leitura do arquivo e criacao do dataframe
    iris_df = pd.read_csv("./data/Iris_Data.csv")

    #separar features da classificacao
    X = iris_df.iloc[:, :4].values #transformar o df num array numpy para funcionar na metrica
    Y = iris_df.iloc[:, 4]

    #cria uma amostra aleatoria
    sample = np.random.rand(1,4)*6

    #classifica a amostra com o algoritmo knn
    print 'sample ', sample, ': ', knn(sample, X, Y, 11)[0]


if __name__ == "__main__":
    main()
