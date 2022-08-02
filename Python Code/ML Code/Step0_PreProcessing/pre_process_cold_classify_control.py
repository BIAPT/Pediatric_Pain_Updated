# Pediatric Pain Project- Machine Learning Models for Analysis 1
# Created by Liz Teel on 08/11/2021
# Edited from Yacine (original files in BIAPT GitHUB Pediatric Pain- Milestone 3)

# Pre-processing script to help load the data correctly
# For full dataset: use pre_process_acute_pain_baseline_vs_cold_control
# For subset (i.e. dPLI only, PE only, etc.): use pre_process_acute_pain_baseline_vs_cold_control_subset

import pandas as pd


def pre_process_acute_pain_baseline_vs_cold_control(input_filename):
    """This function loads, reshapes and cleans the data frame so that it is amenable for machine learning
        Args:
            input_filename (string): This is the path to the dataset saved as a .csv file (DF_FILE_PATH in Step1_LoadCleanData.py)
        Returns:
            X: the features for this data in the form of a matrix (EEG variables)
            y: the label vector for the data which in this analysis is 0 or 1 (dependent variable/what you are trying to classify)
            p_id: the participant ID variable, used for Leave-One-Subject-Out (LOSO) cross validation
    """

    # this pre_processing script is for classifying baseline vs. hot condition
    # MSK and Control groups run separately
    # this code is for the MSK group
    # deleting variables from the dataset that are not necessary for this comparison

    # Read the CSV
    df = pd.read_csv(input_filename)  # import dataset and call it "df"

    # this analysis runs each group separately
    # this is for the controls (1), so dropping pain group (0)
    df = df.drop(df[df.group == 0].index)

    # delete hot conditions (2), since we are only interested in baseline vs cold for this model
    df = df.drop(df[df.state == 2].index)

    # Extract the features for ml analysis (i.e. all columns except for id, group, and state)
    X = df.drop(['id', 'group', 'state', 'baseline'], axis=1).to_numpy()  # axis=1 indicates we are dropping columns from dataset

    y = df.state.to_numpy()  # create separate vector, y, with the binary classification variable (i.e. state)
    group = df.id.to_numpy()  # create separate vector, p_id, with the participant's ID value for leave one subject out CV

    return X, y, group, df  # output matrices for features (X), classication variable (y), participant id (p_id), and entire dataframe (df)

def pre_process_acute_pain_baseline_vs_cold_control_subset(input_filename):
    """This function loads, reshapes and cleans the data frame so that it is amenable for machine learning
        Args:
            input_filename (string): This is the path to the dataset saved as a .csv file (DF_FILE_PATH in Step1_LoadCleanData.py)
        Returns:
            X: the features for this data in the form of a matrix (EEG variables)
            y: the label vector for the data which in this analysis is 0 or 1 (dependent variable/what you are trying to classify)
            p_id: the participant ID variable, used for Leave-One-Subject-Out (LOSO) cross validation
    """

    # this pre_processing script is for classifying baseline vs. hot condition
    # MSK and Control groups run separately
    # this code is for the MSK group
    # deleting variables from the dataset that are not necessary for this comparison

    # Read the CSV
    df = input_filename

    # this analysis runs each group separately
    # this is for the controls (1), so dropping pain group (0)
    df = df.drop(df[df.group == 0].index)

    # delete hot conditions (2), since we are only interested in baseline vs cold for this model
    df = df.drop(df[df.state == 2].index)

    # Extract the features for ml analysis (i.e. all columns except for id, group, and state)
    X = df.drop(['id', 'group', 'state', 'baseline'], axis=1).to_numpy()  # axis=1 indicates we are dropping columns from dataset

    y = df.state.to_numpy()  # create separate vector, y, with the binary classification variable (i.e. state)
    group = df.id.to_numpy()  # create separate vector, p_id, with the participant's ID value for leave one subject out CV

    return X, y, group, df  # output matrices for features (X), classication variable (y), participant id (p_id), and entire dataframe (d