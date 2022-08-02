"""
Pediatric Pain Project- Machine Learning Models
Created by Liz Teel on Jan 26, 2022

This code splits the full project dataset into a training/test set and a validation set.
The validation set is a randomly selected 20% subset of the full dataset for each analysis.
The validation set created here is to be kept separate/not analyzed until the final model is selected.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# File and Dir Path Config
in_path = "/Users/elizabethteel/Desktop/Research/Projects/Musculoskeletal Pain/Datasets/Full Datasets/"
out_path = "/Users/elizabethteel/Desktop/Research/Projects/Musculoskeletal Pain/Datasets/Train and Test Splits/"

""" ORIGINAL DATASET:
EEG FEATURES INCLUDE: PEAK POWER, PE, DPLI, & BINARY GRAPH OUTCOMES
Note: The random state option ensure that we keep the same participants in each test/validation set
for all of the different comparisons"""

# read in full dataset (all groups, all conditions)
full_df = pd.read_csv(in_path + 'pediatricpain_machinelearningdataset_coldonly_nocovas.csv')

# Binary Classification: Baseline vs. Cold in Controls
# reduce datasets to the specific analyses being run
cold_control_df = full_df.drop(full_df[full_df.group == 0].index)  # drop control group

# split dataset 80%/20% into training/test datasets
subjects_cold_control_df = cold_control_df.id.unique()
cold_control_train, cold_control_test = train_test_split(subjects_cold_control_df, random_state=456, test_size=0.2)
cold_control_train_df = full_df[full_df.id.isin(cold_control_train)]
cold_control_valid_df = full_df[full_df.id.isin(cold_control_test)]

# export new training/test sets as csv files
cold_control_train_df.to_csv(out_path + 'cold_control_training_covas.csv', index=False)
cold_control_valid_df.to_csv(out_path + 'cold_control_validation_covas.csv', index=False)

# Binary Classification: Baseline vs. Cold in Pain Patients
# reduce datasets to the specific analyses being run
cold_pain_df = full_df.drop(full_df[full_df.group == 1].index)  # drop pain group

# split dataset 80%/20% into training/test datasets
subjects_cold_pain_df = cold_pain_df.id.unique()
cold_pain_train, cold_pain_test = train_test_split(subjects_cold_pain_df, random_state=567, test_size=0.2)
cold_pain_train_df = full_df[full_df.id.isin(cold_pain_train)]
cold_pain_valid_df = full_df[full_df.id.isin(cold_pain_test)]

# export new training/test sets as csv files
cold_pain_train_df.to_csv(out_path + 'cold_pain_training_covas.csv', index=False)
cold_pain_valid_df.to_csv(out_path + 'cold_pain_validation_covas.csv', index=False)
