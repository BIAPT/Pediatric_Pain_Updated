"""CODE TO CALCULATE THE MODEL ACCURACY USING INDIVIDUAL ELECTRODES- PE THETA OUTCOMES
Control Subjects

Based on the importance of PE features and the theta frequency band identified previously, along with the fact that
PE can be calculated on indivdiual electrodes, we re-ran the best models on PE theta features for each individual
electrode, as this may have more potential for translation in the future. """

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


training_path = "/Users/elizabethteel/Desktop/Research/Projects/Musculoskeletal Pain/Datasets/Train and Test Splits/"
validation_path = "/Users/elizabethteel/Desktop/Research/Projects/Musculoskeletal Pain/Datasets/Train and Test Splits/Validation Sets/"
out_path = "/Users/elizabethteel/Desktop/Research/Projects/Musculoskeletal Pain/Results/Machine Learning/Validation/Single Electrodes/"

# load the datasets
training_filename = training_path + "cold_control_training_covas.csv"
validation_filename = validation_path + "cold_control_validation_covas.csv"

df_train = pd.read_csv(training_filename)
df_valid = pd.read_csv(validation_filename)

# PE outcomes (theta band) only
# drop all other EEG features and frequency bands
df_train = df_train.drop(df_train.filter(regex='peak').columns, axis=1)
df_train = df_train.drop(df_train.filter(regex='dpli').columns, axis=1)
df_train = df_train.drop(df_train.filter(regex='binary').columns, axis=1)
df_train = df_train.drop(df_train.filter(regex='delta').columns, axis=1)
df_train = df_train.drop(df_train.filter(regex='alpha').columns, axis=1)
df_train = df_train.drop(df_train.filter(regex='beta').columns, axis=1)

X_train = df_train.drop(['id', 'group', 'state', 'baseline'], axis=1).to_numpy()  # create x dataset with EEG features only
y_train = df_train.state.to_numpy()  # create separate vector, y, with the binary classification variable (i.e. group)

# repeat above steps for the validation dataset
df_valid = df_valid.drop(df_valid.filter(regex='peak').columns, axis=1)
df_valid = df_valid.drop(df_valid.filter(regex='dpli').columns, axis=1)
df_valid = df_valid.drop(df_valid.filter(regex='binary').columns, axis=1)
df_valid = df_valid.drop(df_valid.filter(regex='delta').columns, axis=1)
df_valid = df_valid.drop(df_valid.filter(regex='alpha').columns, axis=1)
df_valid = df_valid.drop(df_valid.filter(regex='beta').columns, axis=1)

X_valid = df_valid.drop(['id', 'group', 'state', 'baseline'], axis=1).to_numpy()  # axis=1 indicates we are dropping columns from dataset
y_valid = df_valid.state.to_numpy()  # create separate vector, y, with the binary classification variable (i.e. group)

df_headers = df_train.columns.values.tolist()
del df_headers[0:4]  # get PE column headers- add back into feature importance array

# Logistic Regression Models- Controls
print(f"Logistic Regression Model: C=0.1")

# set up the processing pipeline with the model and "cleaning" steps
clf = LogisticRegression(max_iter=10000, penalty='l1', solver='saga', C=0.1)
pipe = Pipeline([
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
    ('scaler', StandardScaler()),
    ('CLF', clf)])

accuracies = []  # creating a blank matrix, where model accuracies will eventually be stored
f1s = []
cms = np.zeros((2, 2))  # creating a blank 2x2 tables, where confusion matrix output will be stored

n = len(X_train[0])
num_folds= len(X_train[0])

for i in range(n):  # loop through each electrode
    print(f"Number of electrodes left: {num_folds}")

    X_train_electrode = [row[i] for row in X_train]  # get only the column for a single electrode
    X_train_electrode = np.array(X_train_electrode)  # convert list back into np array
    X_train_electrode = X_train_electrode[..., None]  # convert into a column format

    X_valid_electrode = [row[i] for row in X_valid]
    X_valid_electrode = np.array(X_valid_electrode)  # convert list back into np array
    X_valid_electrode = X_valid_electrode[..., None]  # convert into a column format

    pipe.fit(X_train_electrode, y_train)  # fit model specified above/learn data using the training sets
    y_hat = pipe.predict(X_valid_electrode)  # using the model parameters that we learned above, use test set X data to predict y

    accuracy = accuracy_score(y_valid, y_hat)  # calculate accuracy of each electrode
    f1 = f1_score(y_valid, y_hat)  # calculate f1 score of each electrode
    cm = confusion_matrix(y_valid, y_hat)  # calculate confusion matrix of each electrode

    # add model performance metrics to a list
    accuracies.append(accuracy)
    f1s.append(f1)
    cms = np.add(cms, cm)

    num_folds = num_folds - 1

clf_data = {
    'accuracies': accuracies,
    'f1s': f1s,
    'cms': cms,
}

# we also want to add the overall classification accuracy for all PE theta features (include all electrodes)
# re-running the code for the full 19-channels
pipe.fit(X_train, y_train) #fit model specified above/learn data using the training sets
y_hat = pipe.predict(X_valid) #using the model parameters that we learned above, use test set X data to predict y

all_accuracy = accuracy_score(y_valid, y_hat)  # calculate full headset accuracy
all_f1 = f1_score(y_valid, y_hat)  # calculate full headset f1 score
all_cm = confusion_matrix(y_valid, y_hat)  # calculate full headset confusion matrix

accuracy_df = pd.DataFrame(accuracies)  # add individual electrode accuracies to a dataframe
accuracy_df = accuracy_df.T
accuracy_df.columns = df_headers  # add column names
accuracy_df['All'] = all_accuracy  # add new column with full headset accuracy

filename = out_path + 'control_logisitic_pe_theta.csv'
accuracy_df.to_csv(filename, index=False)  # save file

# REPEAT PROCESS ABOVE FOR SVM MODEL
# SVM Models- Controls
print(f"SVM Model: k=RBF, C=0.9")

clf = SVC(max_iter=10000, kernel='rbf', C=0.9)
pipe = Pipeline([
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
    ('scaler', StandardScaler()),
    ('CLF', clf)])

accuracies_svm = []  # creating a blank matrix, where model accuracies will eventually be stored
f1s_svm = []
cms_svm = np.zeros((2, 2))  # creating a blank 2x2 tables, where confusion matrix output will be stored

n = len(X_train[0])
num_folds= len(X_train[0])

for i in range(n):
    print(f"Number of electrodes left: {num_folds}")

    X_train_electrode = [row[i] for row in X_train]  # get only the column for a single electrode
    X_train_electrode = np.array(X_train_electrode)  # convert list back into np array
    X_train_electrode = X_train_electrode[..., None]  # convert into a column format

    X_valid_electrode = [row[i] for row in X_valid]
    X_valid_electrode = np.array(X_valid_electrode)  # convert list back into np array
    X_valid_electrode = X_valid_electrode[..., None]  # convert into a column format

    pipe.fit(X_train_electrode, y_train) #fit model (that is specified in gridsearch function above)/learn data using the training sets
    y_hat = pipe.predict(X_valid_electrode) #using the model parameters that we learned above, use test set X data to predict y

    accuracy = accuracy_score(y_valid, y_hat)
    f1 = f1_score(y_valid, y_hat)
    cm = confusion_matrix(y_valid, y_hat)

    accuracies_svm.append(accuracy)
    f1s_svm.append(f1)
    cms_svm = np.add(cms, cm)

    num_folds = num_folds - 1

clf_data = {
    'accuracies': accuracies_svm,
    'f1s': f1s_svm,
    'cms': cms_svm,
}

pipe.fit(X_train, y_train) #fit model (that is specified in gridsearch function above)/learn data using the training sets
y_hat = pipe.predict(X_valid) #using the model parameters that we learned above, use test set X data to predict y

all_accuracy = accuracy_score(y_valid, y_hat)
all_f1 = f1_score(y_valid, y_hat)
all_cm = confusion_matrix(y_valid, y_hat)

accuracy_svm_df = pd.DataFrame(accuracies_svm)
accuracy_svm_df = accuracy_svm_df.T
accuracy_svm_df.columns = df_headers
accuracy_svm_df['All'] = all_accuracy

filename = out_path + 'control_svm_pe_theta.csv'
accuracy_svm_df.to_csv(filename, index=False)
