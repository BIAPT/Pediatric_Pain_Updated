import pandas as pd
import numpy as np
import pickle

import mne

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

in_path = "/Users/elizabethteel/Desktop/Research/Projects/Musculoskeletal Pain/Results/Machine Learning/Validation/Single Electrodes/"

log_filename = in_path + "control_logisitic_pe_theta.csv"
log_model_weights = pd.read_csv(log_filename)
log_model_weights = log_model_weights.multiply(100)

svm_filename = in_path + "control_svm_pe_theta.csv"
svm_model_weights = pd.read_csv(svm_filename)
svm_model_weights = svm_model_weights.multiply(100)

"""CREATING VISUALIZATION SPECIFIC TO THETA PERMUTATION ENTROPY OUTCOMES
Creating topo maps and corresponding bar chart to visual the permutation entropy 
features in the theta frequency band"""

# LOGISTIC REGRESSION MODEL

# create df with absolute value of regression cofficients
pe_theta_log = log_model_weights.T
# set index name based on the electrode order in the dataset
pe_theta_log.index = ['Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8', 'Cz', 'C3', 'C4', 'T3', 'T4', 'T5', 'T6', 'Pz', 'P3', 'P4', 'O1', 'O2', 'All']
# change the order of variables to match with the MNE electrode order
pe_ordered_log = pe_theta_log.reindex(['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4', 'O1', 'O2', 'T3', 'T5', 'T4', 'T6', 'All'])
pe_theta_array_log = pe_ordered_log.drop('All')  # drop the all electrode accuracy for the topomap
pe_theta_array_log = pe_theta_array_log.to_numpy()

full_channels = ['Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8', 'Cz', 'C3', 'C4', 'T3', 'T4', 'T5', 'T6', 'Pz', 'P3', 'P4', 'O1', 'O2']

# Form the 10-20 montage for the DSI-24 System
mont1020 = mne.channels.make_standard_montage('standard_1020')
# Choose what channels you want to keep
# Make sure that these channels exist e.g. T1 does not exist in the standard 10-20 EEG system!
ind = [i for (i, channel) in enumerate(mont1020.ch_names) if channel in full_channels]
mont1020_new = mont1020.copy()
# Keep only the desired channels
mont1020_new.ch_names = [mont1020.ch_names[x] for x in ind]
full_channel_info = [mont1020.dig[x+3] for x in ind]
# Keep the first three rows as they are the fiducial points information
mont1020_new.dig = mont1020.dig[0:3]+full_channel_info
#mont1020_new.plot()

dsi_info = mne.create_info(ch_names=mont1020_new.ch_names, sfreq=300.,
                            ch_types='eeg')

dsi_evoked = mne.EvokedArray(pe_theta_array_log, dsi_info)
dsi_evoked.set_montage(mont1020_new)

# get color maps for all accuracies and save them in a list for bar chart
cmap_list = []
n= 20
pe_ordered_log.columns = ['Accuracy']
ordered_list = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4', 'O1', 'O2', 'T3', 'T5', 'T4', 'T6', 'All']

for i in range(n):
    electrode = ordered_list[i]
    datapoint = pe_ordered_log.at[electrode, 'Accuracy']

    # since our y-axis is from 0.4 to 0.8, we need to convert the datapoint to match this scale
    data_convert = (datapoint-40)/40

    cmap = matplotlib.cm.get_cmap('plasma')  # set the color map of the image you are trying to match
    rgba = cmap(data_convert)  # get the exact color (rgba format) of a specified value/data point on the color map
    hex = matplotlib.colors.rgb2hex(rgba)
    hex_code = f"{hex}"
    #print(f"'{matplotlib.colors.rgb2hex(rgba)}'")  # convert rgba format to hex value

    cmap_list.append(hex_code)

# first creating the topomap and saving it
fig, ax = plt.subplots()
im,cm = mne.viz.plot_topomap(dsi_evoked.data[:, 0], dsi_evoked.info, show=False,vmin=40,vmax=80, cmap='plasma')
plt.xlabel('Classification Accuracy (%)', fontweight='bold', labelpad=15)
# manually fiddle the position of colorbar
ax_x_start = 0.85
ax_x_width = 0.04
ax_y_start = 0.1
ax_y_height = 0.82
cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
clb = fig.colorbar(im, cax=cbar_ax)
plt.savefig(in_path + 'control_logistic_topomap.png')

#topomap = plt.imread(in_path + 'control_logistic_topomap.png')

fig, ax = plt.subplots()
# create bar chart for the PE features
ax.set_xticks(range(len(ordered_list)))
ax.set_xticklabels(ordered_list)
ax.bar(range(len(ordered_list)), pe_ordered_log['Accuracy'], color=cmap_list, edgecolor='black')
ax.hlines(y=50, xmin=-1, xmax=20, color='black')
ax.hlines(y=60, xmin=-1, xmax=20, color='black', linestyles='dashed')
ax.hlines(y=70, xmin=-1, xmax=20, color='black', linestyles='dashdot')
plt.xlabel('Electrodes', fontweight='bold')
plt.ylabel('Classification Accuracy (%)', fontweight='bold')
ax.axis(ymin=40,ymax=80)
plt.savefig(in_path + 'control_logistic_barchart.png')

#bar_graph = plt.imread(in_path + 'control_logistic_barchart.png')

#fig, axarr = plt.subplots(1, 2)
#fig.suptitle('Classification Accuracy (Logistic Regression) for Permutation Entropy (Theta Frequency) Features', fontweight='bold', fontsize=12)
#fig.patch.set_visible(False)

#axarr[0].imshow(topomap)
#axarr[1].imshow(bar_graph)

# SVM MODEL

# create df with absolute value of regression cofficients
pe_theta_log = svm_model_weights.T
# set index name based on the electrode order in the dataset
pe_theta_log.index = ['Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8', 'Cz', 'C3', 'C4', 'T3', 'T4', 'T5', 'T6', 'Pz', 'P3', 'P4', 'O1', 'O2', 'All']
# change the order of variables to match with the MNE electrode order
pe_ordered_log = pe_theta_log.reindex(['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4', 'O1', 'O2', 'T3', 'T5', 'T4', 'T6', 'All'])
pe_theta_array_log = pe_ordered_log.drop('All')  # drop the all electrode accuracy for the topomap
pe_theta_array_log = pe_theta_array_log.to_numpy()

full_channels = ['Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8', 'Cz', 'C3', 'C4', 'T3', 'T4', 'T5', 'T6', 'Pz', 'P3', 'P4', 'O1', 'O2']

# Form the 10-20 montage for the DSI-24 System
mont1020 = mne.channels.make_standard_montage('standard_1020')
# Choose what channels you want to keep
# Make sure that these channels exist e.g. T1 does not exist in the standard 10-20 EEG system!
ind = [i for (i, channel) in enumerate(mont1020.ch_names) if channel in full_channels]
mont1020_new = mont1020.copy()
# Keep only the desired channels
mont1020_new.ch_names = [mont1020.ch_names[x] for x in ind]
full_channel_info = [mont1020.dig[x+3] for x in ind]
# Keep the first three rows as they are the fiducial points information
mont1020_new.dig = mont1020.dig[0:3]+full_channel_info
#mont1020_new.plot()

dsi_info = mne.create_info(ch_names=mont1020_new.ch_names, sfreq=300.,
                            ch_types='eeg')

dsi_evoked = mne.EvokedArray(pe_theta_array_log, dsi_info)
dsi_evoked.set_montage(mont1020_new)

# get color maps for all accuracies and save them in a list for bar chart
cmap_list = []
n= 20
pe_ordered_log.columns = ['Accuracy']
ordered_list = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4', 'O1', 'O2', 'T3', 'T5', 'T4', 'T6', 'All']

for i in range(n):
    electrode = ordered_list[i]
    datapoint = pe_ordered_log.at[electrode, 'Accuracy']

    # since our y-axis is from 0.4 to 0.8, we need to convert the datapoint to match this scale
    data_convert = (datapoint-40)/40

    cmap = matplotlib.cm.get_cmap('plasma')  # set the color map of the image you are trying to match
    rgba = cmap(data_convert)  # get the exact color (rgba format) of a specified value/data point on the color map
    hex = matplotlib.colors.rgb2hex(rgba)
    hex_code = f"{hex}"
    #print(f"'{matplotlib.colors.rgb2hex(rgba)}'")  # convert rgba format to hex value

    cmap_list.append(hex_code)

# first creating the topomap and saving it
fig, ax = plt.subplots()
im,cm = mne.viz.plot_topomap(dsi_evoked.data[:, 0], dsi_evoked.info, show=False,vmin=40,vmax=80, cmap='plasma')
plt.xlabel('Classification Accuracy (%)', fontweight='bold', labelpad=15)
# manually fiddle the position of colorbar
ax_x_start = 0.85
ax_x_width = 0.04
ax_y_start = 0.1
ax_y_height = 0.82
cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
clb = fig.colorbar(im, cax=cbar_ax)
plt.savefig(in_path + 'control_svm_topomap.png')

#topomap = plt.imread(in_path + 'control_svm_topomap.png')

fig, ax = plt.subplots()
# create bar chart for the PE features
ax.set_xticks(range(len(ordered_list)))
ax.set_xticklabels(ordered_list)
ax.bar(range(len(ordered_list)), pe_ordered_log['Accuracy'], color=cmap_list, edgecolor='black')
ax.hlines(y=50, xmin=-1, xmax=20, color='black')
ax.hlines(y=60, xmin=-1, xmax=20, color='black', linestyles='dashed')
ax.hlines(y=70, xmin=-1, xmax=20, color='black', linestyles='dashdot')
plt.xlabel('Electrodes', fontweight='bold')
plt.ylabel('Classification Accuracy (%)', fontweight='bold')
ax.axis(ymin=40,ymax=80)
plt.savefig(in_path + 'control_svm_barchart.png')

#bar_graph = plt.imread(in_path + 'control_svm_barchart.png')

#fig, axarr = plt.subplots(1, 2)
#fig.suptitle('Classification Accuracy (Logistic Regression) for Permutation Entropy (Theta Frequency) Features', fontweight='bold', fontsize=12)
#fig.patch.set_visible(False)

#axarr[0].imshow(topomap)
#axarr[1].imshow(bar_graph)




