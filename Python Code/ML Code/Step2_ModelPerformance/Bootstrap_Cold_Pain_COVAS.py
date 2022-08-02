"""STEP 2a- RUN THE BOOTSTRAPPING PROCEDURE TO GENERATE THE 95% CONFIDENCE INTERVAL FOR THE MODEL
Code for the Pain Group

This code is used to determine the 95% confidence interval of model accuracy. Sampling with replacement is used
to create a new dataframe of equal size, with model accuracy re-calculated on the bootstrapped datatframe.
This procedure is repeated N (in our case, 10,000) time to create a distrubtion of bootstrapped accuracies.
The confidence interval is generated from this distribution.

Line by line annotations are provided in "Bootstrap_Cold_Control_COVAS" for a more detailed description of code"""

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from ml_tools.Step0_DefineModelsandTests import bootstrap_interval
from ml_tools.pre_process_cold_classify_pain import pre_process_acute_pain_baseline_vs_cold_pain

path= "/Users/elizabethteel/Desktop/Research/Projects/Musculoskeletal Pain/Datasets/Train and Test Splits/"
out_path = f'/Users/elizabethteel/Desktop/Research/Projects/Musculoskeletal Pain/Results/Machine Learning/Model Selection- COVAS/Bootstrap/'

filename = path + "cold_pain_training_covas.csv"

# Logistic Regression Models- Controls
boot_filename = out_path + "cold_pain_logisitc_bootstrap.csv"
print("Bootstrap Interval for Baseline vs. Cold (Pain Subjects): Logistic Regression: C=0.1")

boot_data = pd.DataFrame(np.zeros((1, 5)))
names = ['Comparison', 'Acc_Dist Mean', 'Acc_Dist Std', 'Acc_Int_Low', 'Acc_Int_High']
boot_data.columns = names
c = 0

clf = LogisticRegression(max_iter=10000, penalty='l1', solver='saga', C=0.1)
pipe = Pipeline([
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
    ('scaler', StandardScaler()),
    ('CLF', clf)])

X, y, group, df = pre_process_acute_pain_baseline_vs_cold_pain(filename)
acc_distribution, acc_interval = bootstrap_interval(X, y, group, pipe, num_resample=10000, p_value=0.05)

# Print out some high level summary
print("Accuracy Distribution:")
print(acc_distribution)
print(f"Mean: {np.mean(acc_distribution)} and std: {np.std(acc_distribution)}")
print("Bootstrap Interval")
print(acc_interval)

boot_data.loc[c, 'Comparison'] = 'Cold Control'
boot_data.loc[c, 'Acc_Dist Mean'] = np.mean(acc_distribution)
boot_data.loc[c, 'Acc_Dist Std'] = np.std(acc_distribution)
boot_data.loc[c, 'Acc_Int_Low'] = acc_interval[0]
boot_data.loc[c, 'Acc_Int_High'] = acc_interval[1]

boot_data.to_csv(boot_filename, index=False, header=True, sep=',')
print('finished')

# SVM Models- Controls
boot_filename = out_path + "cold_pain_svm_bootstrap.csv"
print("Bootstrap Interval for Baseline vs. Cold (Pain Subjects): SVM Model: C=0.9, Kernel=RBF")

boot_data = pd.DataFrame(np.zeros((1, 5)))
names = ['Comparison', 'Acc_Dist Mean', 'Acc_Dist Std', 'Acc_Int_Low', 'Acc_Int_High']
boot_data.columns = names
c = 0

clf = SVC(max_iter=10000, kernel='rbf', C=0.9)
pipe = Pipeline([
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
    ('scaler', StandardScaler()),
    ('CLF', clf)])

X, y, group, df = pre_process_acute_pain_baseline_vs_cold_pain(filename)
acc_distribution, acc_interval = bootstrap_interval(X, y, group, pipe, num_resample=10000, p_value=0.05)

# Print out some high level summary
print("Accuracy Distribution:")
print(acc_distribution)
print(f"Mean: {np.mean(acc_distribution)} and std: {np.std(acc_distribution)}")
print("Bootstrap Interval")
print(acc_interval)

boot_data.loc[c, 'Comparison'] = 'Cold Control'
boot_data.loc[c, 'Acc_Dist Mean'] = np.mean(acc_distribution)
boot_data.loc[c, 'Acc_Dist Std'] = np.std(acc_distribution)
boot_data.loc[c, 'Acc_Int_Low'] = acc_interval[0]
boot_data.loc[c, 'Acc_Int_High'] = acc_interval[1]

boot_data.to_csv(boot_filename, index=False, header=True, sep=',')
print('finished')

