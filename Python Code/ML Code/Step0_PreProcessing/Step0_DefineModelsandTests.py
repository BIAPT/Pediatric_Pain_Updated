#Pediatric Pain Project- Machine Learning Models for Analysis 1 (Catherine/Don Paper)
#Created by Liz Teel on 10/29/2021
#Edited from Yacine (original files in BIAPT GitHUB Pediatric Pain- Milestone 3)

#Classification script- defines the functions that will be called in the analysis script

#import all the python packages needed to run the functions
from math import floor
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import permutation_test_score
from sklearn.utils import resample

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import joblib

import pickle

import multiprocessing as mp
import os
import sys


def classify_loso(X, y, p_id, clf):
    """ Main classification function to train and test a ml model with Leave one subject out
        Args:
            X (numpy matrix): this is the feature matrix with row being a data point
            y (numpy vector): this is the label vector with row belonging to a data point
            p_id (numpy vector): this is the p_id vector (which is a the participant id)
            clf (sklearn classifier): this is a classifier made in sklearn with fit, transform and predict functionality
        Returns:
            accuracies (list): the accuracies for each leave one out participant
            cms (list): the confusion matrix for each leave one out participant
    """
    logo = LeaveOneGroupOut() #calling the leave one group out (in this case, one subject instead of group) function from sklearn

    accuracies = [] #creating a blank matrix, where model accuracies will eventually be stored
    f1s = []
    cms = np.zeros((2, 2)) #creating a blank 2x2 tables, where confusion matrix (true pos, false pos, etc.) output will be stored

    num_folds = logo.get_n_splits(X, y, p_id)  # keep track of how many folds left
    for train_index, test_index in logo.split(X, y, p_id): #load in feature matrix (X), target matrix (y), and subject matrix (p_id) and split them
        X_train, X_test = X[train_index], X[test_index] #split such that all rows for a single subject becomes the test set
        y_train, y_test = y[train_index], y[test_index] #while all rows for the remaining subjects becomes the training set

        print(f"Number of folds left: {num_folds}")

        with joblib.parallel_backend('loky'): #allows models to run concurrently/parallelized; loky is default
            clf.fit(X_train, y_train) #fit model (that is specified in gridsearch function below)/learn data using the training sets
        y_hat = clf.predict(X_test) #using the model parameters that we learned above, use test set X data to predict y

        accuracy = accuracy_score(y_test, y_hat)
        f1 = f1_score(y_test, y_hat)
        cm = confusion_matrix(y_test, y_hat)

        accuracies.append(accuracy)
        f1s.append(f1)
        cms = np.add(cms, cm)

        num_folds = num_folds - 1

    return accuracies, f1s, cms #output model performance outcomes (accuracy and confusion matrix values)

def classify_loso_importance(X, y, p_id, clf):
    """ Main classification function to train and test a ml model with Leave one subject out
        Args:
            X (numpy matrix): this is the feature matrix with row being a data point
            y (numpy vector): this is the label vector with row belonging to a data point
            p_id (numpy vector): this is the p_id vector (which is a the participant id)
            clf (sklearn classifier): this is a classifier made in sklearn with fit, transform and predict functionality
        Returns:
            accuracies (list): the accuracies for each leave one out participant
            cms (list): the confusion matrix for each leave one out participant
    """
    logo = LeaveOneGroupOut() #calling the leave one group out (in this case, one subject instead of group) function from sklearn

    feature_importance = pd.DataFrame(np.zeros((1, 192))) #creating a blank matrix, where model accuracies will eventually be stored

    num_folds = logo.get_n_splits(X, y, p_id)  # keep track of how many folds left
    for train_index, test_index in logo.split(X, y, p_id): #load in feature matrix (X), target matrix (y), and subject matrix (p_id) and split them
        X_train, X_test = X[train_index], X[test_index] #split such that all rows for a single subject becomes the test set
        y_train, y_test = y[train_index], y[test_index] #while all rows for the remaining subjects becomes the training set

        print(f"Number of folds left: {num_folds}")

        with joblib.parallel_backend('loky'): #allows models to run concurrently/parallelized; loky is default
            clf.fit(X_train, y_train) #fit model (that is specified in gridsearch function below)/learn data using the training sets
        y_hat = clf.predict(X_test) #using the model parameters that we learned above, use test set X data to predict y

        importance = clf.coef_[0]
        feature_importance = pd.concat([feature_importance, importance])

    return feature_importance #output model performance outcomes (accuracy and confusion matrix values)

def permutation_test(X, y, p_id, clf, num_permutation=1000):
    """ Support function to determine if model accuracy is significantly better than chance
        Args:
            X (numpy matrix): this is the feature matrix with row being a data point
            y (numpy vector): this is the label vector with row belonging to a data point
            p_id (numpy vector): this is the p_id vector (which is a the participant id)
            clf (sklearn classifier): this is model being tested (see gridsearch function for specific models)
            num_permutation (int): the number of time to permute y
            random_state (int): this is used for reproducible output
        Returns:
            f1s (list): the f1 at for each leave one out participant

        Permutes targets (Y) to generate ‘randomized data’ and compute the empirical p-value against the null hypothesis that
        features and targets are independent.

        The p-value represents the fraction of randomized data sets where the estimator
        performed as well or better than in the original data. A small p-value suggests that there is a real dependency
        between features and targets which has been used by the estimator to give good predictions. A large p-value may be
        due to lack of real dependency between features and targets or the estimator was not able to use the dependency to
        give good predictions.
    """

    #use LOSO cross validation and split data into test and training sets
    logo = LeaveOneGroupOut()
    train_test_splits = logo.split(X, y, p_id)

    #calculate p-value using sklearn permutation test function
    with joblib.parallel_backend('loky'): #run procedure in parallel
        (accuracies, permutation_scores, p_value) = permutation_test_score(clf, X, y, groups=p_id,
                                                                           cv=train_test_splits,
                                                                           n_permutations=num_permutation,
                                                                           verbose=num_permutation, n_jobs=-1)
        #clf is model specified in gridsearch, n_jobs=1 is default

    return accuracies, permutation_scores, p_value #output accuracies, permutation test scores, and the p-value

def save_model(gs, model_file):
    model_file = open(model_file, 'ab')

    pickle.dump(gs, model_file)
    model_file.close()


def load_pickle(filename):
    '''Helper function to unpickle the pickled python obj'''
    file = open(filename, 'rb')
    data = pickle.load(file)
    file.close()

    return data


def bootstrap_classify(X, y, p_id, clf, sample_id):
    print("Bootstrap sample #" + str(sample_id))
    sys.stdout.flush()  # This is needed when we use multiprocessing

    # Get the sampled with replacement dataset
    sample_X, sample_y, sample_p_id = resample(X, y, p_id)

    # Classify and get the results
    #accuracies, cms = classify_loso(sample_X, sample_y, sample_p_id, clf)
    accuracies, f1s, cms = classify_loso(sample_X, sample_y, sample_p_id, clf)

    return np.mean(accuracies)


def bootstrap_interval(X, y, p_id, clf, num_resample=1000, p_value=0.05):
    """Create a confidence interval for the classifier with the given p value
        Args:
            X (numpy matrix): The feature matrix with which we want to train on classifier on
            y (numpy vector): The label for each row of data point
            p_id (numpy vector): The p_id id for each row in the data (correspond to the participant ids)
            clf (sklearn classifier): The classifier that we which to train and validate with bootstrap interval
            num_resample (int): The number of resample we want to do to create our distribution
            p_value (float): The p values for the upper and lower bound
        Returns:
            f1_distribution (float vector): the distribution of all the f1s
            f1_interval (float vector): a lower and upper interval on the f1s corresponding to the p value
    """

    # Setup the pool of available cores
    #ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK', default=1))
    #pool = mp.Pool(processes=4)

    # Calculate each round asynchronously
    #results = [pool.apply_async(bootstrap_classify, args=(X, y, p_id, clf, sample_id,)) for sample_id in
               #range(num_resample)]

    results = [bootstrap_classify(X, y, p_id, clf, sample_id) for sample_id in range(num_resample)]

    # Unpack the results
    acc_distribution = results

    # Sort the results
    acc_distribution.sort()

    # Set the confidence interval at the right index
    lower_index = floor(num_resample * (p_value / 2))
    upper_index = floor(num_resample * (1 - (p_value / 2)))
    acc_interval = acc_distribution[lower_index], acc_distribution[upper_index]

    return acc_distribution, acc_interval