"""STEP 3a- RUN THE BOOTSTRAPPING PROCEDURE TO GENERATE THE 95% CONFIDENCE INTERVAL FOR THE VALIDATION SET
Code for the Control Group

This code is largely the same as step 2, but creates a 95% CI for the model performance as evaluated on the
validation/hold-out set as opposed to the test train set."""

import pandas as pd
import numpy as np

from math import floor

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.utils import resample

from ml_tools.pre_process_cold_classify_control import pre_process_acute_pain_baseline_vs_cold_control

training_path = "/Users/elizabethteel/Desktop/Research/Projects/Musculoskeletal Pain/Datasets/Train and Test Splits/"
validation_path = "/Users/elizabethteel/Desktop/Research/Projects/Musculoskeletal Pain/Datasets/Train and Test Splits/Validation Sets/"
out_path = "/Users/elizabethteel/Desktop/Research/Projects/Musculoskeletal Pain/Results/Machine Learning/Validation/Bootstrap/"

training_filename = training_path + "cold_control_training_covas.csv"  # get the training set
validation_filename = validation_path + "cold_control_validation_covas.csv"  # get the validation set

# Logistic Regression Models- Controls
print(f"Logistic Regression Model: C=0.1")

# load the model and associated parameters
clf = LogisticRegression(max_iter=10000, penalty='l1', solver='saga', C=0.1)
pipe = Pipeline([
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
    ('scaler', StandardScaler()),
    ('CLF', clf)])

X_train, y_train, group_train, df_train = pre_process_acute_pain_baseline_vs_cold_control(training_filename)  # split the test/train set
X_valid, y_valid, group_valid, df_valid = pre_process_acute_pain_baseline_vs_cold_control(validation_filename)  # spli the validation set

accuracies = []  # creating a blank matrix, where model accuracies will eventually be stored

n = 10000  # number of iterations for the bootstrapping
p_value = 0.05  # value for the CI (0.05 will give you a 95% CI)

num_folds= 10000 # number of iterations for the bootstrapping- using this in the for loop

for i in range(n):
    print(f"Number of folds left: {num_folds}")

    # Get the sampled with replacement dataset
    sample_X, sample_y, sample_p_id = resample(X_train, y_train, group_train)

    pipe.fit(sample_X, sample_y)  # fit model with resampled data
    y_hat = pipe.predict(X_valid)  # using the model parameters learned above, use valid set X data to predict valid y
    accuracy = accuracy_score(y_valid, y_hat)  # calculate accuracy for true valid y and predicted valid y

    accuracies.append(accuracy)
    num_folds = num_folds - 1

accuracies.sort() # Sort the results

# Set the confidence interval at the right index
lower_index = floor(n * (p_value / 2))
upper_index = floor(n * (1 - (p_value / 2)))
acc_interval = accuracies[lower_index], accuracies[upper_index]

# Save the bootstrapped data
bootstrap_df = pd.DataFrame(acc_interval)
bootstrap_df = bootstrap_df.rename(columns={0: "95% CI"})
bootstrap_df = bootstrap_df.rename(index={0: "Lower Bound", 1: "Upper Bound"})
bootstrap_df.to_csv(out_path + 'cold_control_logistic_all_bootstrap.csv')

# REPEAT FOR THE SVM MODEL
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

    # Get the sampled with replacement dataset
    sample_X, sample_y, sample_p_id = resample(X_train, y_train, group_train)

    pipe.fit(sample_X, sample_y)  # fit model with resampled data
    y_hat = pipe.predict(X_valid)  # using the model parameters learned above, use valid set X data to predict valid y
    accuracy = accuracy_score(y_valid, y_hat)  # calculate accuracy for true valid y and predicted valid y

    accuracies_svm.append(accuracy)
    num_folds = num_folds - 1

accuracies_svm.sort()  # Sort the results

# Set the confidence interval at the right index
lower_index = floor(n * (p_value / 2))
upper_index = floor(n * (1 - (p_value / 2)))
acc_interval_svm = accuracies_svm[lower_index], accuracies_svm[upper_index]

bootstrap_df = pd.DataFrame(acc_interval_svm)
bootstrap_df = bootstrap_df.rename(columns={0: "95% CI"})
bootstrap_df = bootstrap_df.rename(index={0: "Lower Bound", 1: "Upper Bound"})
bootstrap_df.to_csv(out_path + 'cold_control_svm_all_bootstrap.csv')


