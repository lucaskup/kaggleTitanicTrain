# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # First Notebook for Kaggle Competition
#
# This notebook is was used in two Kaggle competitions, in the classic Titanic competition
# and in the April Tabular Series Competition. It did not achieve the best results, but got a
# solid 0.77511 acc (people argue that the best achieveble acc in this competition is ~0.83)
# an it scored 0.78449 in the april competition (the first place got 0.81).
#
# TLDR: The datafile is read, we preprocess the data and input nan values using the mean,
# onehotencode the categorical features and use an ensemble of LDA, Random Forest, kNN, SVM and MLP classifiers

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


# %%
# Read the datasets

datasetTrain = pd.read_csv('data/train.csv')
datasetTest = pd.read_csv('data/test.csv')
datasetTrain.head()


# %%
# Defines the function that preprocess the data


def preprocessData(df: pd.DataFrame,
                   columnTransformer: ColumnTransformer = None,
                   complementarySetForNull: pd.DataFrame = None) -> (np.ndarray, np.ndarray, ColumnTransformer):
    '''
    Preprocess the data passed in, fill None values, encodes categorical features.

    Parameters:
        df (DataFrame): A pandas dataframe that contains the data to be preprocessed
        columnTransformer (ColumnTransformer): optional column transformer to be used if null it creates a new ct
        complementarySetForNull (Dataframe): An optional df (test) that is used for inputation of missing values.

    Returns:
        X (Numpy Array): The feature array already preprocessed
        Y (Numpy Array): The target value
        ct (Column Transformer): The column transformer used in the preprocess 
    '''
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
# Preprocess the train and the test set and frees the memory of regarding
# the features of the dataset that were not used

X, Y, ct = preprocessData(datasetTrain, complementarySetForNull=datasetTest)

X_test, _, _ = preprocessData(datasetTest, columnTransformer=ct)
passengerIdTest = datasetTest['PassengerId'].values.reshape(-1, 1)

datasetTrain = None
datasetTest = None
ensembleOfModels = []

# %% [markdown]
# # If you don't have enough hardware...
# Sometimes you just dont have enough hardware available, it was my case in april's competition.
# My solution was to do a grid search using just a sample of the training set. Usually you want to use
# all the data, but if you are on a budget station (like me) a sample can do the job.
# Use the SAMPLE_RATIO (0,1] below to control how much of the training sample you will use.

# %%
# Since we dont have enough hardware to grid search through the entire data
# we will take a i.i.d. sample of 10% of our dataset and will use that
# to grid search and obtain the best hyperparameters for our models

SAMPLE_RATIO = 1
sampleIndexes = np.random.choice(len(Y),
                                 int(len(Y)*SAMPLE_RATIO),
                                 replace=False)
X_sample = X[sampleIndexes]
Y_sample = Y[sampleIndexes]

# %% [markdown]
# # Grid Search
# Some models have hyperparameters, aka values you have to manually set and that are not subject of optimization during the training phase.
# For those, the best practice is to search through some combination of parameters (trial and error) and select the parameters that create the
# best model. In sklearn, we have the GridSearchCV that allows us to do a grid search in a k-fold cross-validation setup, we will do that.

# %%
# Grid Search Through the Random Forest Classifier

gridParameters = {'n_estimators': [10, 50, 100, 200],
                  'criterion': ['gini', 'entropy']}
gsCV = GridSearchCV(RandomForestClassifier(),
                    gridParameters,
                    cv=10,
                    n_jobs=-1)

gsCV.fit(X_sample, Y_sample)

print(
    f'Best Random Forst Classifier:\n   Score > {gsCV.best_score_}\n   Params > {gsCV.best_params_}')
ensembleOfModels.append(gsCV.best_estimator_)


# %%

# Grid Search Through the kNN classifier
# to use mahalonobis distance we need to pass the keyword parameters
# V and VI
# in case we want to know the valid distance metrics
# we could run => sorted(VALID_METRICS['brute'])

covParam = np.cov(X.astype(np.float32))
invCovParam = np.linalg.pinv(covParam)

gridParameters = [{'algorithm': ['auto'],
                  'metric': ['minkowski'],
                   'n_neighbors': [5, 10, 20]},
                  {'algorithm': ['brute'],
                   'metric': ['mahalanobis'],
                   'n_neighbors': [5, 10, 20],
                   'metric_params': [{'V': covParam,
                                      'VI': invCovParam}]}]
gsCV = GridSearchCV(KNeighborsClassifier(),
                    gridParameters,
                    cv=10,
                    n_jobs=-1)

gsCV.fit(X_sample, Y_sample)

print(
    f'Best kNN Classifier:\n   Score > {gsCV.best_score_}\n   Params > {gsCV.best_params_}')
ensembleOfModels.append(gsCV.best_estimator_)


# %%
# The LDA models does not have much hyperparameters to tune

model = LinearDiscriminantAnalysis()
cv = cross_validate(model, X_sample, Y_sample, scoring='accuracy', cv=10)

print(np.mean(cv['test_score']))

ensembleOfModels.append(model)


# %%
# Lets grid search through the SVM classifier

gridParameters = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': ['auto'],  # [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf']}
gsCV = GridSearchCV(svm.SVC(),
                    gridParameters,
                    cv=10,
                    n_jobs=-1)

gsCV.fit(X_sample, Y_sample)

print(
    f'Best SVM:\n   Score > {gsCV.best_score_}\n   Params > {gsCV.best_params_}')
ensembleOfModels.append(gsCV.best_estimator_)


# %%
# Lets grid search through the MLP classifier

gridParameters = {'hidden_layer_sizes': [(5, 5), (10, 5), (10, 10), (15, 10), (15, 15)],
                  'activation': ['logistic', 'relu'],
                  'solver': ['adam'],
                  'alpha': [0.0001, 0.05, 0.005],
                  'learning_rate': ['constant', 'adaptive'],
                  }
gsCV = GridSearchCV(MLPClassifier(max_iter=2500),
                    gridParameters,
                    cv=10,
                    n_jobs=-1)

gsCV.fit(X_sample, Y_sample)

print(
    f'Best MLP:\n   Score > {gsCV.best_score_}\n   Params > {gsCV.best_params_}')
ensembleOfModels.append(gsCV.best_estimator_)

# %% [markdown]
# # The prediction...
# We now take the best model for each of the five algorithms we used (LDA, kNN, Random Forest, SVM) and we train them with selected
# hyperparameters using the whole training set.
#
# We use each of the five classifiers to make a prediction regarding the target variable. If three or more classifiers vote
# on a specific autcome that is the autcome our ensemble classifier.

# %%
# Prepare data for Kaggle Submission
ensembleOfModels = list(map(lambda m: m.fit(X, Y), ensembleOfModels))


predictions = list(map(lambda m: m.predict(X_test), ensembleOfModels))

predictionsEnsemble = predictions[0] + predictions[1] + \
    predictions[2] + predictions[3] + predictions[4]
# predictionsEnsemble = predictionsEnsemble.apply(lambda x: 1 if x >= 3 else 0)
dataForSubmission = pd.DataFrame(np.concatenate((passengerIdTest,
                                                 predictionsEnsemble.reshape(-1, 1)), axis=1), columns=['PassengerId', 'Survived'])
dataForSubmission['Survived'] = dataForSubmission['Survived'].apply(
    lambda x: 1 if x >= 3 else 0)


# %%
# Creates the submission file
dataForSubmission.to_csv('submission/TabularTitanic.csv',
                         sep=',',
                         decimal='.',
                         index=False)
dataForSubmission.head()


# %%
