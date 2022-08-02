"""STEP 3b- RUN THE PERMUTATION PROCEDURE TO GENERATE THE STATISTIC SIGNIFICANCE (P-VALUE) OF THE VALIDATION SET
Code for the Control Group

This code is largely the same as step 2, but determines statistical significnat for the model performance as evaluated
on the validation/hold-out set as opposed to the test train set."""

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

from ml_tools.pre_process_cold_classify_control import pre_process_acute_pain_baseline_vs_cold_control

training_path = "/Users/elizabethteel/Desktop/Research/Projects/Musculoskeletal Pain/Datasets/Train and Test Splits/"
validation_path = "/Users/elizabethteel/Desktop/Research/Projects/Musculoskeletal Pain/Datasets/Train and Test Splits/Validation Sets/"
out_path = "/Users/elizabethteel/Desktop/Research/Projects/Musculoskeletal Pain/Results/Machine Learning/Validation/Permutation/"

training_filename = training_path + "cold_control_training_covas.csv"
validation_filename = validation_path + "cold_control_validation_covas.csv"

# Logistic Regression Models- Controls
print(f"Logistic Regression Model: C=0.1")

# load the model and associated parameters
clf = LogisticRegression(max_iter=10000, penalty='l1', solver='saga', C=0.1)
pipe = Pipeline([
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
    ('scaler', StandardScaler()),
    ('CLF', clf)])

X_train, y_train, group_train, df_train = pre_process_acute_pain_baseline_vs_cold_control(training_filename)  # split the test/train set
X_valid, y_valid, group_valid, df_valid = pre_process_acute_pain_baseline_vs_cold_control(validation_filename)  # split the validation set

accuracies = []  # creating a blank matrix, where model accuracies will eventually be stored

n = 10000  # number of times to create a permutated dataset
num_folds= 10000  # number of iterations- used to count down in for loop

for i in range(n):
    print(f"Number of folds left: {num_folds}")

    y_permute = shuffle(y_train)  # shuffle the labels
    permute_count = np.unique(y_permute, return_counts=True)  # ensure the counts remain equal after shuffle
    true_count = np.unique(y_train, return_counts=True)

    # checking that the true accuracy is reproducible
    #pipe.fit(X_train, y_train) #fit model (that is specified in gridsearch function below)/learn data using the training sets
    #y_hat_true = pipe.predict(X_valid) #using the model parameters that we learned above, use test set X data to predict y
    #accuracy_true = accuracy_score(y_valid, y_hat_true)

    pipe.fit(X_train, y_permute) #fit model above to the shuffled labels
    y_hat_permute = pipe.predict(X_valid) #using the model above, predict y
    accuracy_permute = accuracy_score(y_valid, y_hat_permute)  # evaulate accuracy based on shuffled labels

    accuracies.append(accuracy_permute)
    num_folds = num_folds - 1

# load permutated accuracies into file and save
permutation_df = pd.DataFrame(accuracies)
permutation_df = permutation_df.rename(columns={0: "Random Accuracy"})
permutation_df.to_csv(out_path + 'cold_control_logistic_all_permutation.csv')

# REPEAT ABOVE STEPS FOR THE SVM MODEL
# SVM Models- Controls
print(f"SVM Model: C=0.8, Kernel= RBF")

clf = SVC(max_iter=10000, kernel='rbf', C=0.8)
pipe = Pipeline([
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
    ('scaler', StandardScaler()),
    ('CLF', clf)])

accuracies_svm = []  # creating a blank matrix, where model accuracies will eventually be stored
num_folds= 10000

for i in range(n):
    print(f"Number of folds left: {num_folds}")
    y_permute = shuffle(y_train)

    pipe.fit(X_train, y_permute) #fit model (that is specified in gridsearch function below)/learn data using the training sets
    y_hat_permute = pipe.predict(X_valid) #using the model parameters that we learned above, use test set X data to predict y
    accuracy_permute = accuracy_score(y_valid, y_hat_permute)

    accuracies_svm.append(accuracy_permute)
    num_folds = num_folds - 1

permutation_df = pd.DataFrame(accuracies_svm)
permutation_df = permutation_df.rename(columns={0: "Random Accuracy"})
permutation_df.to_csv(out_path + 'cold_control_svm_all_permutation.csv')

