"""STEP 3- RUN THE BEST MODEL IDENTIFIED IN STEP 1 ON THE VALIDATION SET
Code for the Control Group

This code is largely the same as step 2, but evaluated model performance on the validation/hold-out set as opposed to
the test train set. This code also saves the model weights for the logistric regression model (can't obtain model
weights for SVM when using a rbf kernel), to look at feature importance in future steps.

Line by line annotations are provided in the "Validation_Best_Model_Cold_Pain" for more detail."""

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

from ml_tools.Step0_DefineModelsandTests import classify_loso
from ml_tools.pre_process_cold_classify_control import pre_process_acute_pain_baseline_vs_cold_control

training_path = "/Users/elizabethteel/Desktop/Research/Projects/Musculoskeletal Pain/Datasets/Train and Test Splits/"
validation_path = "/Users/elizabethteel/Desktop/Research/Projects/Musculoskeletal Pain/Datasets/Train and Test Splits/Validation Sets/"
out_path = "/Users/elizabethteel/Desktop/Research/Projects/Musculoskeletal Pain/Results/Machine Learning/Validation/Cold Control/"

training_filename = training_path + "cold_control_training_covas.csv"
validation_filename = validation_path + "cold_control_validation_covas.csv"

# Logistic Regression Models- Controls
control_perform_filename_log = out_path + f"cold_control_best_logistic_all.pickle"
control_weights_filename_log = out_path + f"cold_control_best_logistic_modelweights.sav"
print(f"Logistic Regression Model: C=0.1")

clf = LogisticRegression(max_iter=10000, penalty='l1', solver='saga', C=0.1)
pipe = Pipeline([
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
    ('scaler', StandardScaler()),
    ('CLF', clf)])

X_train, y_train, group_train, df_train = pre_process_acute_pain_baseline_vs_cold_control(training_filename)
X_valid, y_valid, group_valid, df_valid = pre_process_acute_pain_baseline_vs_cold_control(validation_filename)

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

logistic_model_weights = pipe.named_steps['CLF'].coef_
pickle.dump(logistic_model_weights, open(control_weights_filename_log, 'wb'))

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

# SVM Models- Controls
control_perform_filename_svm = out_path + f"cold_control_best_svm_all.pickle"
print(f"SVM Model: C=0.8, Kernel= RBF")

clf = SVC(max_iter=10000, kernel='rbf', C=0.8)
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
print(sum(accuracies))