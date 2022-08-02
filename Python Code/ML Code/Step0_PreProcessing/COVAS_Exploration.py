""" STEP 0.5 DETERMINE WHAT EEG FEATURES IN FULL DATASET ARE INFLUENCED BY MOTOR MOVEMENT
This scripts goes through each "class" of EEG features (i.e. power, wPLI, etc.), runs a ttest to determine
if there are differences between participants w/ true resting baseline and participants w/ arbitrary motor
movements during baseline, and plots/visualizes findings (results summary table and histograms). Features
determined to be significantly influenced by motor movement will be removed from the dataset prior to
model selection."""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import researchpy as rp
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import streamlit as st
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from decimal import Decimal

path= "/Users/elizabethteel/Desktop/Research/Projects/Musculoskeletal Pain/Datasets/Full Datasets/"
out_path = "/Users/elizabethteel/Desktop/Research/Projects/Musculoskeletal Pain/Datasets/Train and Test Splits/"

filename = path + "pediatricpain_machinelearningdataset_coldonly.csv"
df = pd.read_csv(filename)  # load dataset

# pull out descriptive variables
desc_df = df[['id','group', 'state', 'window_number', 'baseline']]

# create df for each variable type
# then create average across all channels/frequency bands for that outcomes
power_df = df.filter(regex='power')
power_avg = power_df.mean(axis=1)

peak_df = df.filter(regex='peak')
peak_avg = peak_df.mean(axis=1)

wpli_df = df.filter(regex='wpli')
wpli_avg = wpli_df.mean(axis=1)

dpli_df = df.filter(regex='dpli')
dpli_avg = dpli_df.mean(axis=1)

pe_df = df.filter(regex='norm_pe')
pe_avg = pe_df.mean(axis=1)

binary_df = df.filter(regex='binary')
binary_path_df = binary_df.filter(regex='path')
binary_path_avg = binary_path_df.mean(axis=1)

binary_cluster_df = binary_df.filter(regex='cluster')
binary_cluster_avg = binary_cluster_df.mean(axis=1)

binary_bsw_df = binary_df.filter(regex='bsw')
binary_bsw_avg = binary_bsw_df.mean(axis=1)

binary_mod_df = binary_df.filter(regex='mod')
binary_mod_avg = binary_mod_df.mean(axis=1)

binary_str_df = binary_df.filter(regex='strength')
binary_str_avg = binary_str_df.mean(axis=1)

weighted_df = df.filter(regex='weighted')
weighted_path_df = weighted_df.filter(regex='path')
weighted_path_avg = weighted_path_df.mean(axis=1)

weighted_cluster_df = weighted_df.filter(regex='cluster')
weighted_cluster_avg = weighted_cluster_df.mean(axis=1)

weighted_bsw_df = weighted_df.filter(regex='bsw')
weighted_bsw_avg = weighted_bsw_df.mean(axis=1)

weighted_mod_df = weighted_df.filter(regex='mod')
weighted_mod_avg = weighted_mod_df.mean(axis=1)

weighted_str_df = weighted_df.filter(regex='strength')
weighted_str_avg = weighted_str_df.mean(axis=1)

# concatenate all the averaged data back into a single df
avg_df = pd.concat([desc_df, power_avg, peak_avg, pe_avg, wpli_avg, dpli_avg, binary_path_avg, binary_cluster_avg,
                    binary_mod_avg, binary_bsw_avg, binary_str_avg, weighted_path_avg, weighted_cluster_avg,
                    weighted_mod_avg, weighted_bsw_avg, weighted_str_avg], axis=1)

# all column names into averaged df
avg_df.columns = ['id', 'group', 'state', 'win_num', 'baseline', 'power', 'peak_freq', 'pe', 'wpli', 'dpli',
                  'bin_path', 'bin_cluster', 'bin_mod', 'bin_bsw', 'bin_str', 'wei_path', 'wei_cluster',
                  'wei_mod', 'wei_bsw', 'wei_str']

baseline_avg_df = avg_df[avg_df.state != 3]  # only looking at differences in baseline condition, dropping cold
baseline_avg_df.info()

out_path = "/Users/elizabethteel/Desktop/"

variables = ['power', 'peak_freq', 'pe', 'wpli', 'dpli', 'bin_path', 'bin_cluster', 'bin_mod', 'bin_bsw', 'bin_str',
             'wei_path', 'wei_cluster', 'wei_mod', 'wei_bsw', 'wei_str']

#variables = ['power']

# run through all variables of interest in a loop
with PdfPages(out_path + "COVAS_BaselineComparisons_AllParticipants.pdf") as pdf:  # save output to PDF file
    for v in variables:

        # run t test
        rp.ttest(group1= baseline_avg_df[f'{v}'][baseline_avg_df['baseline'] == 0], group1_name="Resting",
                 group2= baseline_avg_df[f'{v}'][baseline_avg_df['baseline'] == 1], group2_name="COVAS")

        summary, results = rp.ttest(group1= baseline_avg_df[f'{v}'][baseline_avg_df['baseline'] == 0], group1_name="Resting",
                                    group2= baseline_avg_df[f'{v}'][baseline_avg_df['baseline'] == 1], group2_name="COVAS")

        # format the summary table from the t test
        summary.update(summary[['N', 'Mean', 'SD', 'SE', '95% Conf.', 'Interval']].astype(float))
        summary.update(summary[['N', 'Mean', 'SD', 'SE', '95% Conf.', 'Interval']].applymap('{:,.2f}'.format))

        # Create page for the descriptive info/test statistics for each variable
        fig= plt.figure()
        ax1 = plt.table(cellText=summary.values, colLabels=summary.columns, loc='upper center')  # "plot" summary table
        ax1.auto_set_font_size(False)
        ax1.set_fontsize(10)
        ax1[(1, 0)].set_facecolor('#E5FE77') # add colors to table cells
        ax1[(1, 1)].set_facecolor('#E5FE77')
        ax1[(1, 2)].set_facecolor('#E5FE77')
        ax1[(1, 3)].set_facecolor('#E5FE77')
        ax1[(1, 4)].set_facecolor('#E5FE77')
        ax1[(1, 5)].set_facecolor('#E5FE77')
        ax1[(1, 6)].set_facecolor('#E5FE77')
        ax1[(2, 0)].set_facecolor('#C7EAED')
        ax1[(2, 1)].set_facecolor('#C7EAED')
        ax1[(2, 2)].set_facecolor('#C7EAED')
        ax1[(2, 3)].set_facecolor('#C7EAED')
        ax1[(2, 4)].set_facecolor('#C7EAED')
        ax1[(2, 5)].set_facecolor('#C7EAED')
        ax1[(2, 6)].set_facecolor('#C7EAED')

        for (row, col), cell in ax1.get_celld().items():
            if (row == 0):
                cell.set_text_props(fontproperties=FontProperties(weight='bold'))

        table2 = plt.table(cellText=results.values, colLabels=results.columns, loc='center')
        table2.auto_set_font_size(False)
        table2.set_fontsize(10)
        plt.suptitle(f"Test Statistics for {v} by Baseline Condition", fontsize=16, fontweight='bold')
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.box(on=None)

        pdf.savefig(fig)
        plt.close()

        # create (overlay) histogram and output to separate page
        fig2 = plt.figure(figsize= (20, 10))
        ax2 = fig2.add_subplot(111)

        power_resting = plt.hist(baseline_avg_df[f'{v}'][baseline_avg_df['baseline'] == 0], label="Resting",
                                 color='#E5FE77', density=True, bins=50, alpha=0.75)
        power_covas = plt.hist(baseline_avg_df[f'{v}'][baseline_avg_df['baseline'] == 1], label="COVAS",
                               color='#C7EAED', density=True, bins=50, alpha=0.75)

        plt.suptitle(f"Distribution of {v} by Baseline Condition", fontsize= 18, fontweight='bold')
        plt.xlabel(f'{v}', fontsize=16)
        plt.ylabel("Probability density", fontsize= 16)
        plt.show()
        pdf.savefig(fig2)
        plt.close()

