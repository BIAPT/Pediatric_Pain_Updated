"""STEP 2- RUN THE BEST MODEL IDENTIFIED IN STEP 1 ON EACH EEG FEATURE TYPE
Code for the Pain Group

This code is largely the same as step 1, but only focuses on the best models (SVM & logisitic regression) and
hyperparameters identifed in that code. Additionally, this code looks at each type of EEG features as a stand-alone
features (e.g. PE only, wPLI only, etc) as opposed to all features collectively Line by line annotations are provided in
the "Model_Selection_Cold_Control_COVAS" file if interested in understanding what each line of code does."""

import pandas as pd
import numpy as np
import pickle

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from ml_tools.Step0_DefineModelsandTests import classify_loso

path= "/Users/elizabethteel/Desktop/Research/Projects/Musculoskeletal Pain/Datasets/Train and Test Splits/"
out_path = "/Users/elizabethteel/Desktop/Research/Projects/Musculoskeletal Pain/Results/Machine Learning/Model Selection- COVAS/Cold Pain/"

filename = path + "cold_pain_training_covas.csv"
df = pd.read_csv(filename)

# Drop all features except for one for indivdiual feature-type analysis
# In lines 30-33, only the one variable of interest should be greyed out
df = df.drop(df.filter(regex='peak').columns, axis=1)
df = df.drop(df.filter(regex='norm').columns, axis=1)
df = df.drop(df.filter(regex='binary').columns, axis=1)
#df = df.drop(df.filter(regex='dpli').columns, axis=1)

X = df.drop(['id', 'group', 'state', 'baseline'], axis=1).to_numpy()  # axis=1 indicates we are dropping columns from dataset
y = df.state.to_numpy()  # create separate vector, y, with the binary classification variable (i.e. group)
group = df.id.to_numpy()  # create separate vector, p_id, with the participant's ID value for leave one subject out CV

# Logistic Regression Models- Controls
control_perform_filename_log = out_path + f"cold_pain_best_logistic_dpli.pickle"
print(f"Logistic Regression Model: C=0.1")

clf = LogisticRegression(max_iter=10000, penalty='l1', solver='saga', C=0.1)
pipe = Pipeline([
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
    ('scaler', StandardScaler()),
    ('CLF', clf)])

accuracies, f1s, cms = classify_loso(X, y, group, pipe)

clf_data = {
    'accuracies': accuracies,
    'f1s': f1s,
    'cms': cms,
}

control_final_performance_log_file = open(control_perform_filename_log, 'ab')
pickle.dump(clf_data, control_final_performance_log_file)
control_final_performance_log_file.close()
print(sum(accuracies))

# SVM Models- Controls
control_perform_filename_svm = out_path + f"cold_pain_best_svm_dpli.pickle"
print(f"SVM Model: C=0.9, Kernel= RBF")

clf = SVC(max_iter=10000, kernel='rbf', C=0.9)
pipe = Pipeline([
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
    ('scaler', StandardScaler()),
    ('CLF', clf)])

accuracies, f1s, cms = classify_loso(X, y, group, pipe)

clf_data = {
    'accuracies': accuracies,
    'f1s': f1s,
    'cms': cms,
}

control_final_performance_svm_file = open(control_perform_filename_svm, 'ab')
pickle.dump(clf_data, control_final_performance_svm_file)
control_final_performance_svm_file.close()
print(sum(accuracies))