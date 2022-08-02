"""STEP 3- RUN THE BEST MODEL IDENTIFIED IN STEP 1 ON THE VALIDATION SET
Code for the Pain Group

This code is largely the same as step 2, but evaluated model performance on the validation/hold-out set as opposed to
the test train set. This code also saves the model weights for the logistric regression model (can't obtain model
weights for SVM when using a rbf kernel), to look at feature importance in future steps.

Line by line annotations are provided in the "Model_Selection_Cold_Control_COVAS" for more detail."""

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

from ml_tools.pre_process_cold_classify_pain import pre_process_acute_pain_baseline_vs_cold_pain

training_path = "/Users/elizabethteel/Desktop/Research/Projects/Musculoskeletal Pain/Datasets/Train and Test Splits/"
validation_path = "/Users/elizabethteel/Desktop/Research/Projects/Musculoskeletal Pain/Datasets/Train and Test Splits/Validation Sets/"
out_path = "/Users/elizabethteel/Desktop/Research/Projects/Musculoskeletal Pain/Results/Machine Learning/Validation/Cold Pain/"

training_filename = training_path + "cold_pain_training_covas.csv"  # get the test/train set
validation_filename = validation_path + "cold_pain_validation_covas.csv"  # get the validation set

# Logistic Regression Models- Pain
pain_perform_filename_log = out_path + f"cold_pain_best_logistic_all.pickle"  # filename for the validation set performance outcomes
pain_weights_filename_log = out_path + f"cold_pain_best_logistic_modelweights.sav"  # filename for the logistic regression model weights (feature importance)
print(f"Logistic Regression Model: C=0.1")

#  define model parameters
clf = LogisticRegression(max_iter=10000, penalty='l1', solver='saga', C=0.1)
pipe = Pipeline([
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
    ('scaler', StandardScaler()),
    ('CLF', clf)])

X_train, y_train, group_train, df_train = pre_process_acute_pain_baseline_vs_cold_pain(training_filename)  # split the test/train dataset
X_valid, y_valid, group_valid, df_valid = pre_process_acute_pain_baseline_vs_cold_pain(validation_filename)  # split the validation set

accuracies = []  # creating a blank matrix, where model accuracies will eventually be stored
f1s = [] # creating a blank matrix, where f1 scores will eventually be stored
cms = np.zeros((2, 2))  # creating a blank 2x2 tables, where confusion matrix output will be stored

pipe.fit(X_train, y_train) #fit model defined above
y_hat = pipe.predict(X_valid) #using the model parameters that we learned above, use test set X data to predict y

accuracy = accuracy_score(y_valid, y_hat)  # calculate accuracy
print(f"Accuracy: {accuracy}")
f1 = f1_score(y_valid, y_hat)   # calculate f1 score
print(f"F1 Score: {f1}")
cm = confusion_matrix(y_valid, y_hat)   # calculate confusion matrix
print(f"Confusion Matrix: {cm}")

logistic_model_weights = pipe.named_steps['CLF'].coef_  # obtain model weights
pickle.dump(logistic_model_weights, open(pain_weights_filename_log, 'wb'))  # save model weights to pickle file

accuracies.append(accuracy)
f1s.append(f1)
cms = np.add(cms, cm)

# put all performance metrics in a file
clf_data = {
    'accuracies': accuracies,
    'f1s': f1s,
    'cms': cms,
}

# save performance metrics to a pickle file
pain_final_performance_log_file = open(pain_perform_filename_log, 'ab')
pickle.dump(clf_data, pain_final_performance_log_file)
pain_final_performance_log_file.close()

# REPEAT ABOVE STEPS FOR THE SVM MODEL
# SVM Models- Pain
pain_perform_filename_svm = out_path + f"cold_pain_best_svm_all.pickle"
print(f"SVM Model: C=0.9, Kernel=RBF")

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

pain_final_performance_svm_file = open(pain_perform_filename_svm, 'ab')
pickle.dump(clf_data, pain_final_performance_svm_file)
pain_final_performance_svm_file.close()

