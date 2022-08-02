"""STEP 2b- RUN THE PERMUTATION PROCEDURE TO GENERATE THE STATISTIC SIGNIFICANCE (P-VALUE) OF THE MODEL
Code for the Pain Group

This code is used to determine the statistical significance of model accuracy. The y-values (labels) of the dataset
are randomly shuffled and model accuracy is recalculated on the shuffled dataset. This procedure is repeated N
(in our case, 10,000) time to create a distrubtion of permutated accuracies. The p-value is then calculcated
by comparing the true model accuracy to the distribution of the shuffle model accuracies."""

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from ml_tools.Step0_DefineModelsandTests import permutation_test
from ml_tools.pre_process_cold_classify_pain import pre_process_acute_pain_baseline_vs_cold_pain

path= "/Users/elizabethteel/Desktop/Research/Projects/Musculoskeletal Pain/Datasets/Train and Test Splits/"
out_path = f'/Users/elizabethteel/Desktop/Research/Projects/Musculoskeletal Pain/Results/Machine Learning/Model Selection- COVAS/Permutation/'

filename = path + "cold_pain_training_covas.csv"

# Logistic Regression Models- Controls
permutation_filename = out_path + "cold_pain_logisitic_permutation_all.csv"
print("Permutation Test for Baseline vs. Cold (Pain Subjects): Logistic Regression: C=0.1")

perm_data = pd.DataFrame(np.zeros((1, 4)))
names = ['Comparison', 'Random Mean', 'Accuracy', 'p-value']
perm_data.columns = names
c = 0

clf = LogisticRegression(max_iter=10000, penalty='l1', solver='saga', C=0.1)
pipe = Pipeline([
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
    ('scaler', StandardScaler()),
    ('CLF', clf)])

X, y, group, df = pre_process_acute_pain_baseline_vs_cold_pain(filename)
acc, perms, p_value = permutation_test(X, y, group, pipe, num_permutation=10000)

# Print out some high level summary
print("Random:")
print(np.mean(perms))
print("Actual Improvement")
print(acc)
print("P Value:")
print(p_value)

perm_data.loc[c, 'Comparison'] = 'Cold Pain'
perm_data.loc[c, 'Random Mean'] = np.mean(perms)
perm_data.loc[c, 'Accuracy'] = acc
perm_data.loc[c, 'p-value'] = p_value

perm_data.to_csv(permutation_filename, index=False, header=True, sep=',')
print('finished')


# SVM Models- Controls
permutation_filename = out_path + "cold_pain_svm_permutation_all.csv"
print("Permutation Test for Baseline vs. Cold (Pain Subjects): SVM Model: C=0.9, Kernel=RBF")

perm_data = pd.DataFrame(np.zeros((1, 4)))
names = ['Comparison', 'Random Mean', 'Accuracy', 'p-value']
perm_data.columns = names
c = 0

clf = SVC(max_iter=10000, kernel='rbf', C=0.9)
pipe = Pipeline([
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
    ('scaler', StandardScaler()),
    ('CLF', clf)])

X, y, group, df = pre_process_acute_pain_baseline_vs_cold_pain(filename)
acc, perms, p_value = permutation_test(X, y, group, pipe, num_permutation=10000)

# Print out some high level summary
print("Random:")
print(np.mean(perms))
print("Actual Improvement")
print(acc)
print("P Value:")
print(p_value)

perm_data.loc[c, 'Comparison'] = 'Cold Pain'
perm_data.loc[c, 'Random Mean'] = np.mean(perms)
perm_data.loc[c, 'Accuracy'] = acc
perm_data.loc[c, 'p-value'] = p_value

perm_data.to_csv(permutation_filename, index=False, header=True, sep=',')
print('finished')
