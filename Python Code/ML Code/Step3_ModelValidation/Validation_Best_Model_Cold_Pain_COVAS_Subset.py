"""STEP 3- RUN THE BEST MODEL IDENTIFIED IN STEP 1 ON THE VALIDATION SET
Code for the Pain Group

This code is largely the same as step 2, but evaluated model performance on the validation/hold-out set as opposed to
the test train set. This code also saves the model weights for the logistric regression model (can't obtain model
weights for SVM when using a rbf kernel), to look at feature importance in future steps.

Looking at EEG feature types individually (e.g. wPLI only, etc.) as opposed to full feature compliment.

Line by line annotations are provided in the "Validation_Best_Model_Cold_Pain" for more detail."""

import pandas as pd
import numpy as np
import pickle

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

training_path = "/Users/elizabethteel/Desktop/Research/Projects/Musculoskeletal Pain/Datasets/Train and Test Splits/"
validation_path = "/Users/elizabethteel/Desktop/Research/Projects/Musculoskeletal Pain/Datasets/Train and Test Splits/Validation Sets/"
out_path = "/Users/elizabethteel/Desktop/Research/Projects/Musculoskeletal Pain/Results/Machine Learning/Validation/Cold Pain/"

training_filename = training_path + "cold_pain_training_covas.csv"
validation_filename = validation_path + "cold_pain_validation_covas.csv"

df_train = pd.read_csv(training_filename)
df_valid = pd.read_csv(validation_filename)

# Drop all features except for one for indivdiual feature-type analysis
# In lines 39-42, only the one variable of interest should be greyed out
df_train = df_train.drop(df_train.filter(regex='peak').columns, axis=1)
df_train = df_train.drop(df_train.filter(regex='norm').columns, axis=1)
df_train = df_train.drop(df_train.filter(regex='binary').columns, axis=1)
#df_train = df_train.drop(df_train.filter(regex='dpli').columns, axis=1)

X_train = df_train.drop(['id', 'group', 'state', 'baseline'], axis=1).to_numpy()  # axis=1 indicates we are dropping columns from dataset
y_train = df_train.state.to_numpy()  # create separate vector, y, with the binary classification variable (i.e. group)

df_valid = df_valid.drop(df_valid.filter(regex='peak').columns, axis=1)
df_valid = df_valid.drop(df_valid.filter(regex='norm').columns, axis=1)
df_valid = df_valid.drop(df_valid.filter(regex='binary').columns, axis=1)
#df_valid = df_valid.drop(df_valid.filter(regex='dpli').columns, axis=1)

X_valid = df_valid.drop(['id', 'group', 'state', 'baseline'], axis=1).to_numpy()  # axis=1 indicates we are dropping columns from dataset
y_valid = df_valid.state.to_numpy()  # create separate vector, y, with the binary classification variable (i.e. group)

# Logistic Regression Models- Controls
control_perform_filename_log = out_path + f"cold_pain_best_logistic_dpli.pickle"
print(f"Logistic Regression Model: C=0.1")

clf = LogisticRegression(max_iter=10000, penalty='l1', solver='saga', C=0.1)
pipe = Pipeline([
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
    ('scaler', StandardScaler()),
    ('CLF', clf)])

accuracies = []  # creating a blank matrix, where model accuracies will eventually be stored
f1s = []
cms = np.zeros((2, 2))  # creating a blank 2x2 tables, where confusion matrix output will be stored

pipe.fit(X_train, y_train) #fit model (that is specified in gridsearch function below)/learn data using the training sets
y_hat = pipe.predict(X_valid) #using the model parameters that we learned above, use test set X data to predict y

accuracy = accuracy_score(y_valid, y_hat)
print(f"Accuracy: {accuracy}")
f1 = f1_score(y_valid, y_hat)
print(f"F1 Score: {f1}")
cm = confusion_matrix(y_valid, y_hat)
print(f"Confusion Matrix: {cm}")

accuracies.append(accuracy)
f1s.append(f1)
cms = np.add(cms, cm)

clf_data = {
    'accuracies': accuracies,
    'f1s': f1s,
    'cms': cms,
}

control_final_performance_log_file = open(control_perform_filename_log, 'ab')
pickle.dump(clf_data, control_final_performance_log_file)
control_final_performance_log_file.close()

## SVM Models- Controls
control_perform_filename_svm = out_path + f"cold_pain_best_svm_dpli.pickle"
print(f"SVM Model: C=0.9, Kernel= RBF")

clf = SVC(max_iter=10000, kernel='rbf', C=0.9)
pipe = Pipeline([
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
    ('scaler', StandardScaler()),
    ('CLF', clf)])

accuracies = []  # creating a blank matrix, where model accuracies will eventually be stored
f1s = []
cms = np.zeros((2, 2))  # creating a blank 2x2 tables, where confusion matrix output will be stored

pipe.fit(X_train, y_train) #fit model (that is specified in gridsearch function below)/learn data using the training sets
y_hat = pipe.predict(X_valid) #using the model parameters that we learned above, use test set X data to predict y

accuracy = accuracy_score(y_valid, y_hat)
print(f"Accuracy: {accuracy}")
f1 = f1_score(y_valid, y_hat)
print(f"F1 Score: {f1}")
cm = confusion_matrix(y_valid, y_hat)
print(f"Confusion Matrix: {cm}")

accuracies.append(accuracy)
f1s.append(f1)
cms = np.add(cms, cm)

clf_data = {
    'accuracies': accuracies,
    'f1s': f1s,
    'cms': cms,
}

control_final_performance_svm_file = open(control_perform_filename_svm, 'ab')
pickle.dump(clf_data, control_final_performance_svm_file)
control_final_performance_svm_file.close()
