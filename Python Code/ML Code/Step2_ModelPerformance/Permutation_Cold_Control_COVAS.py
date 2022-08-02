"""STEP 2b- RUN THE PERMUTATION PROCEDURE TO GENERATE THE STATISTIC SIGNIFICANCE (P-VALUE) OF THE MODEL
Code for the Control Group

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
from ml_tools.pre_process_cold_classify_control import pre_process_acute_pain_baseline_vs_cold_control_subset

path= "/Users/elizabethteel/Desktop/Research/Projects/Musculoskeletal Pain/Datasets/Train and Test Splits/"

# load data
filename = path + "cold_control_training_covas.csv"
input_file = pd.read_csv(filename)

out_path = f'/Users/elizabethteel/Desktop/Research/Projects/Musculoskeletal Pain/Results/Machine Learning/Model Selection- COVAS/Permutation/'

# Logistic Regression Models- Controls
permutation_filename = out_path + "cold_control_logisitic_permutation.csv"
print("Permutation Test for Baseline vs. Cold (Control Subjects): Logistic Regression: C=0.1")

perm_data = pd.DataFrame(np.zeros((1, 4)))  # define blank dataset to save results of permutation analysis
names = ['Comparison', 'Random Mean', 'Accuracy', 'p-value']  # add column names for new dataset
perm_data.columns = names
c = 0

clf = LogisticRegression(max_iter=10000, penalty='l1', solver='saga', C=0.1)  # define the best logistic regression model defined in Step 1
pipe = Pipeline([
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),  # use the same data "cleaning" procedures defined previously
    ('scaler', StandardScaler()),
    ('CLF', clf)])

X, y, group, df = pre_process_acute_pain_baseline_vs_cold_control_subset(input_file)  # split the dataset as defined previously
acc, perms, p_value = permutation_test(X, y, group, pipe, num_permutation=10000)  # run the permutation procedure
# permutation code is defined in "Step0_DefineModelsandTest"

# Print out some high level summary
print("Random:")
print(np.mean(perms))
print("Actual Improvement")
print(acc)
print("P Value:")
print(p_value)

perm_data.loc[c, 'Comparison'] = 'Cold Control'  # save comparison to permutation results dataframe
perm_data.loc[c, 'Random Mean'] = np.mean(perms)  # save the mean of permutated accuracy
perm_data.loc[c, 'Accuracy'] = acc  # save the accuracy of the real model
perm_data.loc[c, 'p-value'] = p_value  # save the p-value

perm_data.to_csv(permutation_filename, index=False, header=True, sep=',')  # save the full permutation df
print('finished')


# SVM Models- Controls
permutation_filename = out_path + "cold_control_svm_permutation.csv"
print("Permutation Test for Baseline vs. Cold (Control Subjects): SVM Model: C=0.8, Kernel=RBF")

perm_data = pd.DataFrame(np.zeros((1, 4)))
names = ['Comparison', 'Random Mean', 'Accuracy', 'p-value']
perm_data.columns = names
c = 0

clf = SVC(max_iter=10000, kernel='rbf', C=0.8)
pipe = Pipeline([
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
    ('scaler', StandardScaler()),
    ('CLF', clf)])

X, y, group, df = pre_process_acute_pain_baseline_vs_cold_control_subset(input_file)
acc, perms, p_value = permutation_test(X, y, group, pipe, num_permutation=10000)

# Print out some high level summary
print("Random:")
print(np.mean(perms))
print("Actual Improvement")
print(acc)
print("P Value:")
print(p_value)

perm_data.loc[c, 'Comparison'] = 'Cold Control'
perm_data.loc[c, 'Random Mean'] = np.mean(perms)
perm_data.loc[c, 'Accuracy'] = acc
perm_data.loc[c, 'p-value'] = p_value

perm_data.to_csv(permutation_filename, index=False, header=True, sep=',')
print('finished')
