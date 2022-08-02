"""STEP 1- SELECT THE BEST MODEL AND MODEL HYPERPARAMETERS
Code for the Control Group"""

import numpy as np
import pickle

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from ml_tools.Step0_DefineModelsandTests import classify_loso
from ml_tools.pre_process_cold_classify_control import pre_process_acute_pain_baseline_vs_cold_control

path= "/Users/elizabethteel/Desktop/Research/Projects/Musculoskeletal Pain/Datasets/Train and Test Splits/"
out_path = "/Users/elizabethteel/Desktop/Research/Projects/Musculoskeletal Pain/Results/Machine Learning/Model Selection- COVAS/Cold Control/"

# load data
filename = path + "cold_control_training_covas.csv"

# LDA model- Control
control_perform_filename_filename_lda = out_path + 'cold_control_lda.pickle'  # create filename for model performance metrics
print("LDA Model")

clf = LinearDiscriminantAnalysis()  # define model
pipe = Pipeline([
        ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),  # impute missing values with mean
        ('scaler', StandardScaler()),  # scale features
        ('CLF', clf)])  # model statement defined above

X, y, group, df = pre_process_acute_pain_baseline_vs_cold_control(filename)  # split df into x (features), y (label), group (participant ID)
accuracies, f1s, cms = classify_loso(X, y, group, pipe)  # use LOSO CV scheme to determine accuracy, f1score, and confusion matrix

clf_data = {  # load model performance metrics into a file
    'accuracies': accuracies,
    'f1s': f1s,
    'cms': cms,
}

control_final_performance_lda_file = open(control_perform_filename_filename_lda, 'ab')
pickle.dump(clf_data, control_final_performance_lda_file)  # save model performance metrics in pickle file
control_final_performance_lda_file.close()
print(sum(accuracies))

# Logistic Regression Models- Controls
C_iter = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5, 10]  # hyperparameter sweep for C values
for c in C_iter:

    control_perform_filename_log = out_path + f"cold_control_logisticregression_c_{c}.pickle"
    print(f"Logistic Regression Model: C={c}")

    clf = LogisticRegression(max_iter=10000, penalty='l1', solver='saga', C=c)  # logsitic regression model
    pipe = Pipeline([
        ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
        ('scaler', StandardScaler()),
        ('CLF', clf)])

    X, y, group, df = pre_process_acute_pain_baseline_vs_cold_control(filename)
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
kernel = ['linear', 'rbf']  # hyperparameter sweep for SVM kernel
for c in C_iter:  # also sweep through C values defined above
    for k in kernel:

        control_perform_filename_svm = out_path + f"cold_control_svm_kernel_{k}_c_{c}.pickle"
        print(f"SVM Model: C={c}, Kernel={k}")

        clf = SVC(max_iter=10000, kernel=k, C=c)
        pipe = Pipeline([
            ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
            ('scaler', StandardScaler()),
            ('CLF', clf)])

        X, y, group, df = pre_process_acute_pain_baseline_vs_cold_control(filename)
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

# Decision Tree Models- Controls
Crit = ['gini', 'entropy']  # hyperparameter sweep for the decision tree
for cr in Crit:

    control_perform_filename_tree = out_path + f"cold_control_decisiontree_crit_{cr}.pickle"
    print(f"Decision Tree Model: Criterion={cr}")

    clf = DecisionTreeClassifier(criterion=cr)  # decision tree model
    pipe = Pipeline([
        ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
        ('scaler', StandardScaler()),
        ('CLF', clf)])

    X, y, group, df = pre_process_acute_pain_baseline_vs_cold_control(filename)
    accuracies, f1s, cms = classify_loso(X, y, group, pipe)

    clf_data = {
        'accuracies': accuracies,
        'f1s': f1s,
        'cms': cms,
    }

    control_final_performance_tree_file = open(control_perform_filename_tree, 'ab')
    pickle.dump(clf_data, control_final_performance_tree_file)
    control_final_performance_tree_file.close()
    print(sum(accuracies))