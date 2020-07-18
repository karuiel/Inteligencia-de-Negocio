# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 06:12:49 2018
Empieza Lgbm
Fitting 3 folds for each of 117 candidates, totalling 351 fits
[Parallel(n_jobs=1)]: Done 351 out of 351 | elapsed: 65.1min finished
Mejores parámetros grid 1:
{'learning_rate': 0.15, 'n_estimators': 200, 'num_leaves': 80}
Empieza RF
Fitting 3 folds for each of 18 candidates, totalling 54 fits
[Parallel(n_jobs=1)]: Done  54 out of  54 | elapsed: 39.0min finished
Mejores parámetros grid 2:
{'criterion': 'entropy', 'min_samples_leaf': 5, 'n_estimators': 500}

Fitting 5 folds for each of 32 candidates, totalling 160 fits
[Parallel(n_jobs=1)]: Done 160 out of 160 | elapsed: 176.7min finished
Mejores parámetros grid 2:
{'criterion': 'gini', 'min_samples_leaf': 5, 'n_estimators': 300}
@author: Miguemc
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

data_labels=pd.read_csv('water_pump_tra_target.csv')
data_values=pd.read_csv('water_pump_tra.csv')
data = data_values.merge(data_labels, on='id')

data.isnull().sum()
data.population.min()
data['gps_height'].replace(0.0, np.nan, inplace=True)
data['population'].replace(0.0, np.nan, inplace=True)
data['amount_tsh'].replace(0.0, np.nan, inplace=True)
data.isnull().sum()


data.groupby(['region','permit']).size()

data["gps_height"].fillna(data.groupby(['region', 'district_code'])["gps_height"].transform("mean"), inplace=True)
data["gps_height"].fillna(data.groupby(['region'])["gps_height"].transform("mean"), inplace=True)
data["gps_height"].fillna(data["gps_height"].mean(), inplace=True)
data["population"].fillna(data.groupby(['region', 'district_code'])["population"].transform("median"), inplace=True)
data["population"].fillna(data.groupby(['region'])["population"].transform("median"), inplace=True)
data["population"].fillna(data["population"].median(), inplace=True)
data["amount_tsh"].fillna(data.groupby(['region', 'district_code'])["amount_tsh"].transform("median"), inplace=True)
data["amount_tsh"].fillna(data.groupby(['region'])["amount_tsh"].transform("median"), inplace=True)
data["amount_tsh"].fillna(data["amount_tsh"].median(), inplace=True)
data.isnull().sum()

print(data.latitude.max()-data.latitude.min())
print(data.longitude.max()-data.longitude.min())


features=['amount_tsh', 'gps_height', 'population']
scaler = MinMaxScaler(feature_range=(0,20))
data[features] = scaler.fit_transform(data[features])
data[features].head(20)
data.isnull().sum()


plt.figure(figsize=(13,6))
sns.countplot(data.status_group, palette = 'Set3')
data.status_group.value_counts()

plt.figure(figsize=(14,6))
sns.countplot(data=data,x='water_quality',hue='status_group')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
data.water_quality.value_counts()

#looking at regions
plt.figure(figsize=(24, 9))

sns.countplot(data=data,x='region',hue='status_group')



plt.figure(figsize=(24, 9))

sns.countplot(data=data,x='public_meeting',hue='status_group')



(sns
  .FacetGrid(data, 
             hue='status_group',size=14)
  .map(sns.kdeplot, 'population', shade=True)
 .add_legend()
)

(sns
  .FacetGrid(data, 
             hue='status_group',size=10)
  .map(sns.kdeplot, 'construction_year', shade=True)
 .add_legend()
)


(sns
  .FacetGrid(data, 
             hue='status_group',size=10)
  .map(sns.kdeplot, 'gps_height', shade=True)
 .add_legend()
)


(sns
  .FacetGrid(data, 
             hue='status_group',size=10)
  .map(sns.kdeplot, 'latitude', shade=True)
 .add_legend()
)

(sns
  .FacetGrid(data, 
             hue='status_group',size=10)
  .map(sns.kdeplot, 'longitude', shade=True)
 .add_legend()
)


(sns
  .FacetGrid(data, 
             hue='status_group',size=15)
  .map(sns.kdeplot, 'amount_tsh', shade=True)
 .add_legend()
)

plt.figure(figsize=(24, 9))

sns.countplot(data=data,x='source_type',hue='status_group')

plt.figure(figsize=(24, 9))

sns.countplot(data=data,x='extraction_type_group',hue='status_group')


plt.figure(figsize=(24, 9))

sns.countplot(data=data,x='extraction_type_class',hue='status_group')

plt.figure(figsize=(24, 9))

sns.countplot(data=data,x='waterpoint_type',hue='status_group')

plt.figure(figsize=(24, 9))

sns.countplot(data=data,x='quantity',hue='status_group')

data['longitude'].replace(0.0, np.nan, inplace=True)
data['latitude'].replace(0.0, np.nan, inplace=True)
data['construction_year'].replace(0.0, np.nan, inplace=True)
data.groupby(['district_code', 'region','construction_year']).size()


data["latitude"].fillna(data.groupby(['region', 'district_code'])["latitude"].transform("mean"), inplace=True)
data["longitude"].fillna(data.groupby(['region', 'district_code'])["longitude"].transform("mean"), inplace=True)
data["longitude"].fillna(data.groupby(['region'])["longitude"].transform("mean"), inplace=True)
data["construction_year"].fillna(data.groupby(['region', 'district_code'])["construction_year"].transform("median"), inplace=True)
data["construction_year"].fillna(data.groupby(['region'])["construction_year"].transform("median"), inplace=True)
data["construction_year"].fillna(data.groupby(['district_code'])["construction_year"].transform("median"), inplace=True)
data["construction_year"].fillna(data["construction_year"].median(), inplace=True)
print(data.isnull().sum())





