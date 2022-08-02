"""STEP 1- SELECT THE BEST MODEL AND MODEL HYPERPARAMETERS
Code for the Control Group

Notes: line by line annotations are provided in the "Model_Selection_Cold_Control_COVAS" file if needed"""

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
from ml_tools.pre_process_cold_classify_pain import pre_process_acute_pain_baseline_vs_cold_pain

path= "/Users/elizabethteel/Desktop/Research/Projects/Musculoskeletal Pain/Datasets/Train and Test Splits/"
out_path = "/Users/elizabethteel/Desktop/Research/Projects/Musculoskeletal Pain/Results/Machine Learning/Model Selection- COVAS/Cold Pain/"

filename = path + "cold_pain_training_covas.csv"

# LDA model- Pain
pain_perform_filename_filename_lda = out_path + 'cold_pain_lda.pickle'
print("LDA Model")

clf = LinearDiscriminantAnalysis()
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

pain_final_performance_lda_file = open(pain_perform_filename_filename_lda, 'ab')
pickle.dump(clf_data, pain_final_performance_lda_file)
pain_final_performance_lda_file.close()
print(sum(accuracies))

# Logistic Regression Models- Pain
C_iter = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5, 10]
for c in C_iter:

    pain_perform_filename_log = out_path + f"cold_pain_logisticregression_c_{c}.pickle"
    print(f"Logistic Regression Model: C={c}")

    clf = LogisticRegression(max_iter=10000, penalty='l1', solver='saga', C=c)
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
kernel = ['linear', 'rbf']
for c in C_iter:
    for k in kernel:

        pain_perform_filename_svm = out_path + f"cold_pain_svm_kernel_{k}_c_{c}.pickle"
        print(f"SVM Model: C={c}, Kernel={k}")

        clf = SVC(max_iter=10000, kernel=k, C=c)
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

# Decision Tree Models- Pain
Crit = ['gini', 'entropy']
for cr in Crit:

    pain_perform_filename_tree = out_path + f"cold_pain_decisiontree_crit_{cr}.pickle"
    print(f"Decision Tree Model: Criterion={cr}")

    clf = DecisionTreeClassifier(criterion=cr)
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

    pain_final_performance_tree_file = open(pain_perform_filename_tree, 'ab')
    pickle.dump(clf_data, pain_final_performance_tree_file)
    pain_final_performance_tree_file.close()
    print(sum(accuracies))
