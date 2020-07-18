# -*- coding: utf-8 -*-
"""
Autor:
    Jorge Casillas
    Miguel Morales Castillo
Fecha:
    Noviembre/2018
Contenido:
    Modificación del ejemplo de uso de clustering en Python
    Inteligencia de Negocio
    Grado en Ingeniería Informática
    Universidad de Granada
"""

'''
Documentación sobre clustering en Python:
    http://scikit-learn.org/stable/modules/clustering.html
    http://www.learndatasci.com/k-means-clustering-algorithms-python-intro/
    http://hdbscan.readthedocs.io/en/latest/comparing_clustering_algorithms.html
    https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
    http://www.learndatasci.com/k-means-clustering-algorithms-python-intro/
'''

import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import sklearn.cluster as cl
from sklearn import metrics
from sklearn import preprocessing
from math import floor
import seaborn as sns
sns.set()
def norm_to_zero_one(df):
    return (df - df.min()) * 1.0 / (df.max() - df.min())

censo = pd.read_csv('censo_granada.csv')
censo = censo.replace(np.NaN,0) #los valores en blanco realmente son otra categoría que nombramos como 0

#seleccionar casos
caso3 = censo.loc[(censo['ESTUMAD']<5) & (censo["ESTUPAD"]<5)&(censo["EDAD"]<30)]
caso2 = censo.loc[(censo['EDAD']<25)]
caso1 = censo.loc[(censo['NPFAM']>3) & (censo["EDAD"]>30)&(censo["EDAD"]<55)]
#seleccionar variables de interés para clustering
usadas = ['EDAD', 'ANOE', 'ESTHOG', 'NPFAM', 'ESREAL']

X = caso1[usadas]
Y = caso2[usadas]
Z = caso3[usadas]
casos=[X,Y,Z]
X_normal = X.apply(norm_to_zero_one)
Y_normal = Y.apply(norm_to_zero_one)
Z_normal = Z.apply(norm_to_zero_one)

casos_normal=[X_normal,Y_normal,Z_normal]

k_means = cl.KMeans(init='k-means++', n_clusters=4, n_init=5)
dbscan1 = cl.DBSCAN(eps=0.16,min_samples=90)
dbscan2 = cl.DBSCAN(eps=0.15,min_samples=70)
dbscan3 = cl.DBSCAN(eps=0.15,min_samples=70)
dbscan=[dbscan1,dbscan2,dbscan3]
agglomerative = cl.AgglomerativeClustering(n_clusters=4)
birch = cl.Birch(threshold=0.3, n_clusters=4)
mean=cl.MeanShift(cluster_all=True)
mean3=cl.MeanShift(cluster_all=True,bandwidth=0.4)
spectral=cl.SpectralClustering(n_clusters=3)
nombres=["k_means","dbscan","agglomerative","birch","mean"]
algoritmos=[k_means,dbscan, agglomerative, birch, mean]

for j in range(3):
    for i in range(5):
        if (j==2) & (i==4):
            t = time.time()
            print('----- Ejecutando '+nombres[i],end='')
            cluster_predict = mean3.fit_predict(casos_normal[j])
            tiempo = time.time() - t
            
        else:            
            if i==1:
                t = time.time()
                print('----- Ejecutando '+nombres[i],end='')
                cluster_predict = algoritmos[i][j].fit_predict(casos_normal[j])
                tiempo = time.time() - t
            else:
                t = time.time()
                print('----- Ejecutando '+nombres[i],end='')
                cluster_predict = algoritmos[i].fit_predict(casos_normal[j])
                tiempo = time.time() - t
            print(": {:.2f} segundos, ".format(tiempo), end='')
            metric_CH = metrics.calinski_harabaz_score(casos_normal[j], cluster_predict)
            print("Calinski-Harabaz Index: {:.3f}, ".format(metric_CH), end='')
            #el cálculo de Silhouette puede consumir mucha RAM. Si son muchos datos, se puede seleccionar una muestra, p.ej., el 10%
            metric_SC = metrics.silhouette_score(casos_normal[j], cluster_predict, metric='euclidean', sample_size=floor(0.1*len(casos[j])), random_state=123456)
            print("Silhouette Coefficient: {:.5f}".format(metric_SC))
            
            #se convierte la asignación de clusters a DataFrame
            clusters = pd.DataFrame(cluster_predict,index=casos[j].index,columns=['cluster'])
            print("Tamaño de cada cluster:")
            size=clusters['cluster'].value_counts()
            for num,l in size.iteritems():
               print('%s: %5d (%5.2f%%)' % (num,l,100*l/len(clusters)))
            if i!=1 and i!=2 and i!=3:
                centers = pd.DataFrame(algoritmos[i].cluster_centers_,columns=list(casos[j]))
                centers_desnormal = centers.copy()
                
                #se convierten los centros a los rangos originales antes de normalizar
                for var in list(centers):
                    centers_desnormal[var] = casos[j][var].min() + centers[var] * (casos[j][var].max() - casos[j][var].min())
                
                heat= sns.heatmap(centers, cmap="YlGnBu", annot=centers_desnormal, fmt='.3f').get_figure().savefig(nombres[i]+"_"+str(j)+"heat.png")
            
            
            #'''
            print("---------- Preparando el scatter matrix...")
            #se añade la asignación de clusters como columna a casos[j]
            X_algoritmo = pd.concat([casos[j], clusters], axis=1)
            variables = list(X_algoritmo)
            variables.remove('cluster')
            sns_plot = sns.pairplot(X_algoritmo, vars=variables, hue="cluster", palette='Paired', plot_kws={"s": 25}, diag_kind="hist") #en hue indicamos que la columna 'cluster' define los colores
            sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03);
            sns_plot.savefig(nombres[i]+"_"+str(j)+".png")
            plt.close('all')
        #'''
