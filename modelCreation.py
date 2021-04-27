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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn.neural_network import MLPClassifier


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
ensembleOfModels = []
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
ensembleOfModels.append(gsCV.best_estimator_)

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
# sorted(VALID_METRICS['brute'])
gsCV = GridSearchCV(KNeighborsClassifier(),
                    gridParameters,
                    cv=10)

gsCV.fit(X, Y)

print(
    f'Best kNN Classifier:\n   Score > {gsCV.best_score_}\n   Params > {gsCV.best_params_}')
ensembleOfModels.append(gsCV.best_estimator_)
# %%

model = LinearDiscriminantAnalysis()
cv = cross_validate(model, X, Y, scoring='accuracy', cv=10)

print(np.mean(cv['test_score']))

ensembleOfModels.append(model)

# %%

gridParameters = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf']}
gsCV = GridSearchCV(svm.SVC(),
                    gridParameters,
                    cv=10)

gsCV.fit(X, Y)

print(
    f'Best SVM:\n   Score > {gsCV.best_score_}\n   Params > {gsCV.best_params_}')
ensembleOfModels.append(gsCV.best_estimator_)
# %%
gridParameters = {'hidden_layer_sizes': [(5, 5), (10, 5), (15,)],
                  'activation': ['logistic', 'relu'],
                  'solver': ['adam'],
                  'alpha': [0.0001, 0.05],
                  'learning_rate': ['constant', 'adaptive'],
                  }
gsCV = GridSearchCV(MLPClassifier(max_iter=2500),
                    gridParameters,
                    cv=10)

gsCV.fit(X, Y)

print(
    f'Best MLP:\n   Score > {gsCV.best_score_}\n   Params > {gsCV.best_params_}')
ensembleOfModels.append(gsCV.best_estimator_)

# %%
