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

datasetTrain = pd.read_csv('dataCompetition/train.csv')
datasetTest = pd.read_csv('dataCompetition/test.csv')
datasetTrain.head()

# %%


def preprocessData(df,
                   columnTransformer=None,
                   complementarySetForNull=None):
    df.drop(labels=['Name', 'Ticket'],
            axis=1,
            inplace=True)
    if complementarySetForNull is not None:
        # Age fillna with mean age
        completeDF = pd.concat(
            [df, complementarySetForNull]).reset_index(drop=True)
        meanAge = completeDF['Age'].mean()
        meanFare = completeDF['Fare'].mean()
        completeDF = None

        df['Age'] = df['Age'].fillna(meanAge)
        complementarySetForNull['Age'] = complementarySetForNull['Age'].fillna(
            meanAge)

        df['Fare'] = df['Fare'].fillna(meanFare)
        complementarySetForNull['Fare'] = complementarySetForNull['Fare'].fillna(
            meanFare)
    df = df.assign(AgeGroup=df['Age'].apply(
        lambda x: x // 10 if x // 10 <= 6 else 6))
    df = df.assign(hasFamily=(df['SibSp'] + df['Parch']).apply(
        lambda x: 1 if x > 0 else 0))
    df = df.assign(familySize=(df['SibSp'] + df['Parch']).apply(
        lambda x: 0 if pd.isnull(x) else x))
    df = df.assign(hasCabin=df['Cabin'].apply(
        lambda x: 0 if pd.isnull(x) else 1))
    df = df.assign(cabinLetter=df['Cabin'].apply(
        lambda x: '.' if pd.isnull(x) else str(x)[0]))
    df['Fare'] = df['Fare'].apply(
        lambda x: 0 if pd.isnull(x) else x)
    df = df.assign(farePerFamily=df['Fare']/(df['familySize']+1))
    X = df[['Pclass', 'Sex', 'Fare', 'Embarked',
            'AgeGroup', 'hasFamily', 'hasCabin', 'cabinLetter', 'farePerFamily']].values
    if 'Survived' in df.columns:
        Y = np.ravel(df['Survived'])
    else:
        Y = None
    if columnTransformer is None:
        columnTransformer = ColumnTransformer(
            [('encoder', OneHotEncoder(drop='first'), [0, 1, 3, 4, 7]),
             ('minMaxScaler', MinMaxScaler(), [2, 8])], remainder='passthrough')
        X = columnTransformer.fit_transform(X)
    else:
        X = columnTransformer.transform(X)

    return X, Y, columnTransformer


# %%
X, Y, ct = preprocessData(datasetTrain, complementarySetForNull=datasetTest)

X_test, _, _ = preprocessData(datasetTest, columnTransformer=ct)
passengerIdTest = datasetTest['PassengerId'].values.reshape(-1, 1)

datasetTrain = None
datasetTest = None
ensembleOfModels = []
# %%
sampleIndexes = np.random.choice(len(Y), int(len(Y)*0.1), replace=False)

X_sample = X[sampleIndexes]
Y_sample = Y[sampleIndexes]
# %%

# model = RandomForestClassifier(n_estimators=100,
#                               criterion='gini')
gridParameters = {'n_estimators': [10, 50, 100, 125],
                  'criterion': ['gini', 'entropy']}
gsCV = GridSearchCV(RandomForestClassifier(),
                    gridParameters,
                    cv=3,
                    n_jobs=-1)

gsCV.fit(X_sample, Y_sample)

print(
    f'Best Random Forst Classifier:\n   Score > {gsCV.best_score_}\n   Params > {gsCV.best_params_}')
ensembleOfModels.append(gsCV.best_estimator_)

# cv_results = cross_validate(model, X, Y, cv=5,
#                            scoring=['accuracy', 'precision', 'recall'])


# %%
# covParam = np.cov(X.astype(np.float32))
# invCovParam = np.linalg.pinv(covParam)

gridParameters = [{'algorithm': ['auto'],
                  'metric': ['minkowski'],
                   'n_neighbors': [5, 10, 20]}]  # ,
# {'algorithm': ['brute'],
# 'metric': ['mahalanobis'],
# 'n_neighbors': [5, 10],
# 'metric_params': [{'V': covParam,
#                   'VI': invCovParam}]}]
# sorted(VALID_METRICS['brute'])
gsCV = GridSearchCV(KNeighborsClassifier(),
                    gridParameters,
                    cv=3,
                    n_jobs=-1)

gsCV.fit(X_sample, Y_sample)

print(
    f'Best kNN Classifier:\n   Score > {gsCV.best_score_}\n   Params > {gsCV.best_params_}')
ensembleOfModels.append(gsCV.best_estimator_)
# %%

model = LinearDiscriminantAnalysis()
cv = cross_validate(model, X_sample, Y_sample, scoring='accuracy', cv=5)

print(np.mean(cv['test_score']))

ensembleOfModels.append(model)

# %%

gridParameters = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': ['auto'],  # [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf']}
gsCV = GridSearchCV(svm.SVC(),
                    gridParameters,
                    cv=3,
                    n_jobs=-1)

gsCV.fit(X_sample, Y_sample)

print(
    f'Best SVM:\n   Score > {gsCV.best_score_}\n   Params > {gsCV.best_params_}')
ensembleOfModels.append(gsCV.best_estimator_)
# %%
gridParameters = {'hidden_layer_sizes': [(5, 5), (10, 5), (10, 10), (15, 10), (15, 15)],
                  'activation': ['logistic', 'relu'],
                  'solver': ['adam'],
                  'alpha': [0.0001, 0.05],
                  'learning_rate': ['constant', 'adaptive'],
                  }
gsCV = GridSearchCV(MLPClassifier(max_iter=2500),
                    gridParameters,
                    cv=3,
                    n_jobs=-1)

gsCV.fit(X_sample, Y_sample)

print(
    f'Best MLP:\n   Score > {gsCV.best_score_}\n   Params > {gsCV.best_params_}')
ensembleOfModels.append(gsCV.best_estimator_)

# %%
ensembleOfModels2 = list(map(lambda m: m.fit(X, Y), ensembleOfModels))


predictions = list(map(lambda m: m.predict(X_test), ensembleOfModels2))

predictionsEnsemble = predictions[0] + predictions[1] + \
    predictions[2] + predictions[3] + predictions[4]
# predictionsEnsemble = predictionsEnsemble.apply(lambda x: 1 if x >= 3 else 0)
dataForSubmission = pd.DataFrame(np.concatenate((passengerIdTest,
                                                 predictionsEnsemble.reshape(-1, 1)), axis=1), columns=['PassengerId', 'Survived'])
dataForSubmission['Survived'] = dataForSubmission['Survived'].apply(
    lambda x: 1 if x >= 3 else 0)

dataForSubmission.to_csv('submission/TabularPlaygroundApr2021_2.csv',
                         sep=',',
                         decimal='.',
                         index=False)

# %%
