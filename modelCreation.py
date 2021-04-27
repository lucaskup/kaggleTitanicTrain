# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# Import Libraries
from sklearn.neighbors import VALID_METRICS
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.neighbors import DistanceMetric

from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
# import matplotlib.pyplot as plt

# plt.style.use('seaborn-pastel')

# %%
dataset = pd.read_csv('dataCompetition/train.csv')
dataset.head()

# %%
dataset.drop(labels=['Name', 'Ticket'],
             axis=1,
             inplace=True)

dataset = dataset.assign(AgeGroup=dataset['Age'].apply(
    lambda x: x // 10 if x // 10 <= 6 else 6))
dataset = dataset.assign(hasFamily=(dataset['SibSp'] + dataset['Parch']).apply(
    lambda x: 1 if x > 0 else 0))
dataset = dataset.assign(hasCabin=dataset['Cabin'].apply(
    lambda x: 0 if pd.isnull(x) else 1))
dataset['Fare'] = dataset['Fare'].apply(
    lambda x: 0 if pd.isnull(x) else x)

# %%

X = dataset[['Pclass', 'Sex', 'Fare', 'Embarked',
             'AgeGroup', 'hasFamily', 'hasCabin']].values
Y = np.ravel(dataset['Survived'])

# %%
columnTransformer = ColumnTransformer(
    [('encoder', OneHotEncoder(drop='first'), [0, 1, 3, 4]),
     ('minMaxScaler', MinMaxScaler(), [2])], remainder='passthrough')
X = columnTransformer.fit_transform(X)

# %%

# model = RandomForestClassifier(n_estimators=100,
#                               criterion='gini')
gridParameters = {'n_estimators': [10, 50, 100, 200],
                  'criterion': ['gini', 'entropy']}
gsCV = GridSearchCV(RandomForestClassifier(),
                    gridParameters,
                    cv=10)

gsCV.fit(X, Y)

print(
    f'Best Random Forst Classifier:\n   Score > {gsCV.best_score_}\n   Params > {gsCV.best_params_}')


# cv_results = cross_validate(model, X, Y, cv=10,
#                            scoring=['accuracy', 'precision', 'recall'])


# %%
covParam = np.cov(X.astype(np.float32))
invCovParam = np.linalg.pinv(covParam)

gridParameters = [{'algorithm': ['brute'],
                  'metric': ['minkowski'],
                   'n_neighbors': [3, 5, 10]},
                  {'algorithm': ['brute'],
                  'metric': ['mahalanobis'],
                   'n_neighbors': [3, 5, 10],
                   'metric_params': [{'V': covParam,
                                     'VI': invCovParam}]}]

gsCV = GridSearchCV(KNeighborsClassifier(),
                    gridParameters,
                    cv=10)

gsCV.fit(X, Y)

print(
    f'Best kNN Classifier:\n   Score > {gsCV.best_score_}\n   Params > {gsCV.best_params_}')

# %%
