# -*- coding: utf-8 -*-
"""
Autor:
    Jorge Casillas y Miguel Morales Castillo
Fecha:
    Noviembre/2018
Contenido:
    Uso simple de XGB y LightGBM para competir en DrivenData:
       https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/
    Inteligencia de Negocio
    Grado en Ingeniería Informática
    Universidad de Granada
"""

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import Lasso
import xgboost as xgb
import lightgbm as lgb
import datetime
now = datetime.datetime.now()
le = preprocessing.LabelEncoder()

'''
lectura de datos
'''
#los ficheros .csv se han preparado previamente para sustituir ,, y "Not known" por NaN (valores perdidos)
data_x = pd.read_csv('water_pump_tra.csv')
data_y = pd.read_csv('water_pump_tra_target.csv')
data_x_tst = pd.read_csv('water_pump_tst.csv')

#se quitan las columnas que no se usan
data_x.drop(labels=['id'], axis=1,inplace = True)


data_x_tst.drop(labels=['id'], axis=1,inplace = True)


data_y.drop(labels=['id'], axis=1,inplace = True)
    
'''
Se convierten las variables categóricas a variables numéricas (ordinales)
'''
from sklearn.preprocessing import LabelEncoder

mask = data_x.isnull()
data_x_tmp = data_x.fillna(0)
data_x_tmp = data_x_tmp.astype(str).apply(LabelEncoder().fit_transform)
#data_x_nan = data_x_tmp.where(~mask, data_x)
data_x_nan = data_x_tmp

mask = data_x_tst.isnull() #máscara para luego recuperar los NaN
data_x_tmp = data_x_tst.fillna(0) #LabelEncoder no funciona con NaN, se asigna un valor no usado
data_x_tmp = data_x_tmp.astype(str).apply(LabelEncoder().fit_transform) #se convierten categóricas en numéricas
#data_x_tst_nan = data_x_tmp.where(~mask, data_x_tst) #se recuperan los NaN
data_x_tst_nan = data_x_tmp #se recuperan los NaN

#Preprocesamiento avanzado
data_x_nan['date_recorded'] = pd.to_datetime(data_x_nan['date_recorded'])
data_x_nan['operational_year'] = data_x_nan.date_recorded.dt.year - data_x_nan.construction_year
data_x_tst_nan['date_recorded'] = pd.to_datetime(data_x_tst_nan['date_recorded'])
data_x_tst_nan['operational_year'] = data_x_tst_nan.date_recorded.dt.year - data_x_tst_nan.construction_year
useless_features=['date_recorded','wpt_name','num_private','subvillage','region_code','recorded_by','management_group','source_type','source_class','extraction_type_group','extraction_type_class','scheme_name','payment_type','quality_group','quantity_group','waterpoint_type_group','installer','public_meeting','permit']

X = data_x_nan.drop(useless_features,axis=1).values
X_tst = data_x_tst_nan.drop(useless_features,axis=1).values
y = np.ravel(data_y.values)

#------------------------------------------------------------------------
'''
Validación cruzada con particionado estratificado y control de la aleatoriedad fijando la semilla
'''

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123456)
le = preprocessing.LabelEncoder()
def validacion_cruzada(modelo, X, y, cv):
    y_pred_all = []

    for train, test in cv.split(X, y):
        t = time.time()
        modelo = modelo.fit(X[train],y[train])
        tiempo = time.time() - t
        y_pred = modelo.predict(X[test])
        print("Score: {:.4f}, tiempo: {:6.2f} segundos".format(accuracy_score(y[test],y_pred) , tiempo))
        y_pred_all = np.concatenate((y_pred_all,y_pred))
        
    print("")
    return modelo
def train(modelo, X, y):
    clf= validacion_cruzada(modelo,X,y,skf)
    return clf

def test(modelo,X,X_tst,y):
    clf = modelo.fit(X,y)
    y_pred_tra = clf.predict(X)
    y_pred_tst = clf.predict(X_tst)
    df_submission_tst = pd.read_csv('water_pump_submissionformat.csv')
    df_submission_tst['status_group'] = y_pred_tst
    print("Score Train: {:.4f}".format(accuracy_score(y,y_pred_tra)))
    df_submission_tst.to_csv("submission"+clf.__class__.__name__+str(now.strftime("%y-%m-%d-%H-%M"))+".csv", index=False)
    return clf

def test_less_train(modelo,X,X_tst,y):
    y_pred_tst = modelo.predict(X_tst)
    df_submission_tst = pd.read_csv('water_pump_submissionformat.csv')
    df_submission_tst['status_group'] = y_pred_tst
    df_submission_tst.to_csv("submission"+modelo.__class__.__name__+str(now.strftime("%y-%m-%d-%H-%M"))+".csv", index=False)
    return modelo
#------------------------------------------------------------------------



#print(str(X.shape))
#print(str(X_tst.shape))

#clf1 = xgb.XGBClassifier(n_estimators = 1000)
#clf3 = GradientBoostingClassifier(n_estimators=1000)
clf4 = lgb.LGBMClassifier(n_estimators=200,learning_rate=0.15,num_leaves=80)
clf2 = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, criterion='gini')
clf3 = RandomForestClassifier(n_estimators=500, min_samples_leaf=5,criterion='entropy')
eclf= VotingClassifier(estimators=[('RF1', clf3), ('rf2', clf2), ('Lgb', clf4)], voting='hard')
eclf=train(eclf,X,y)
eclf=test(eclf,X,X_tst,y)

#clf = GradientBoostingClassifier(n_estimators=1000,min_samples_leaf=50)
#clf = clf4
#clf=train(clf,X,y)
#clf=test(clf,X,X_tst,y)
'''print("Empieza Lgbm")
params_lgbm = {'learning_rate':[i/100 for i in range(5,70,5) ],'num_leaves':[30,50,80], 'n_estimators':[100,200,500]}
grid1 = GridSearchCV(clf4, params_lgbm, cv=3, n_jobs=1, verbose=1, scoring=make_scorer(accuracy_score))
grid1.fit(X,y)
grid1=test_less_train(grid1,X,X_tst,y)
print("Mejores parámetros grid 1:")
print(grid1.best_params_)


print("Empieza RF")
params_rf = {'n_estimators':[200,500,1000],'min_samples_leaf':[5,30,50], 'criterion':['gini','entropy']}
grid2 = GridSearchCV(clf2, params_rf, cv=3, n_jobs=1, verbose=1, scoring=make_scorer(accuracy_score))
grid2.fit(X,y)
grid2=test_less_train(grid2,X,X_tst,y)
print("Mejores parámetros grid 2:")
print(grid2.best_params_)

print("Empieza RF")
params_rf = {'n_estimators':[200,300,500,700],'min_samples_leaf':[5,15,30,50], 'criterion':['gini','entropy']}
grid2 = GridSearchCV(clf2, params_rf, cv=5, n_jobs=1, verbose=1, scoring=make_scorer(accuracy_score))
grid2.fit(X,y)
grid2=test_less_train(grid2,X,X_tst,y)
print("Mejores parámetros grid 2:")
print(grid2.best_params_)'''
