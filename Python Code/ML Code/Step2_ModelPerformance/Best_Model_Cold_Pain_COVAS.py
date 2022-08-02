"""STEP 2- RUN THE BEST MODEL IDENTIFIED IN STEP 1
Code for the Pain Group

This code is largely the same as step 1, but only focuses on the best models (SVM & logisitic regression) and
hyperparameters identified in that code. Line by line annotations are provided in the
"Model_Selection_Cold_Control_COVAS" file if interested in understanding what each line of code does."""

import numpy as np
import pickle

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from ml_tools.Step0_DefineModelsandTests import classify_loso
from ml_tools.pre_process_cold_classify_pain import pre_process_acute_pain_baseline_vs_cold_pain

path= "/Users/elizabethteel/Desktop/Research/Projects/Musculoskeletal Pain/Datasets/Train and Test Splits/"
out_path = "/Users/elizabethteel/Desktop/Research/Projects/Musculoskeletal Pain/Results/Machine Learning/Model Selection- COVAS/Cold Pain/"

filename = path + "cold_pain_training_covas.csv"

# Logistic Regression Models- Pain
pain_perform_filename_log = out_path + f"cold_pain_best_logistic_all.pickle"
print(f"Logistic Regression Model: C=0.1")

clf = LogisticRegression(max_iter=10000, penalty='l1', solver='saga', C=0.1)
pipe = Pipeline([
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
    ('scaler', StandardScaler()),
    ('CLF', clf)])

X, y, group, df = pre_process_acute_pain_baseline_vs_cold_pain(filename)
accuracies, f1s, cms = classify_loso(X, y, group, pipe)

clf_data = {
    'accuracies': accuracies,
    'f1s': f1s,
    'cms': cms,
}

pain_final_performance_log_file = open(pain_perform_filename_log, 'ab')
pickle.dump(clf_data, pain_final_performance_log_file)
pain_final_performance_log_file.close()
print(sum(accuracies))

# SVM Models- Pain
pain_perform_filename_svm = out_path + f"cold_pain_best_svm_all.pickle"
print(f"SVM Model: C=0.9, Kernel=RBF")

clf = SVC(max_iter=10000, kernel='rbf', C=0.9)
pipe = Pipeline([
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
    ('scaler', StandardScaler()),
    ('CLF', clf)])

X, y, group, df = pre_process_acute_pain_baseline_vs_cold_pain(filename)
accuracies, f1s, cms = classify_loso(X, y, group, pipe)

clf_data = {
    'accuracies': accuracies,
    'f1s': f1s,
    'cms': cms,
}

pain_final_performance_svm_file = open(pain_perform_filename_svm, 'ab')
pickle.dump(clf_data, pain_final_performance_svm_file)
pain_final_performance_svm_file.close()
print(sum(accuracies))
