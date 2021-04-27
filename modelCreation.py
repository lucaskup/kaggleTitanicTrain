# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# Import Libraries
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer

from sklearn.ensemble import RandomForestClassifier

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
Y = np.ravel(dataset['Survived']).reshape(-1, 1)

# %%
columnTransformer = ColumnTransformer(
    [('encoder', OneHotEncoder(drop='first'), [0, 1, 3, 4]),
     ('minMaxScaler', MinMaxScaler(), [2])], remainder='passthrough')
X = columnTransformer.fit_transform(X)

# %%
