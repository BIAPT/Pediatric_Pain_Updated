""" CODE USED TO GENERATE MANUSCRIPT FIGURE 1 (PAIN RESULTS)

Line by line annotation available in the "Validation_Visualize_ImportantFeatures_Control" for more detail"""

import pandas as pd
import pickle

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


in_path = "/Users/elizabethteel/Desktop/Research/Projects/Musculoskeletal Pain/Results/Machine Learning/Validation/Cold Pain/"
data_path= "/Users/elizabethteel/Desktop/Research/Projects/Musculoskeletal Pain/Datasets/Train and Test Splits/Validation Sets/"

header_filename = data_path + "cold_pain_validation_covas.csv"
df = pd.read_csv(header_filename)

df_headers = df.columns.values.tolist()
del df_headers[0:4]  # get column headers- add back into feature importance array

filename = in_path + "cold_pain_best_logistic_modelweights.sav"
model_weights = pickle.load(open(filename, 'rb'))

importance_df = pd.DataFrame(model_weights)

importance_long = importance_df.T
importance_long.columns = ['model_coef']
importance_long['feature'] = df_headers

importance_long['Freq'] = pd.np.where(importance_long.feature.str.contains("delta"), "delta",
                   		  pd.np.where(importance_long.feature.str.contains("theta"), "theta",
                   		  pd.np.where(importance_long.feature.str.contains("alpha"), "alpha", "beta")))

importance_long['Type'] = pd.np.where(importance_long.feature.str.contains("peak"), "peak",
                   		  pd.np.where(importance_long.feature.str.contains("dpli"), "dpli",
                   		  pd.np.where(importance_long.feature.str.contains("norm"), "pe", "graph")))


importance_long['model_ab'] = importance_long['model_coef'].abs()
importance_small = importance_long.nlargest(19, 'model_ab')


importance_small['Electrode'] = pd.np.where(importance_small.feature.str.contains("fp2"), "Fp2",
								pd.np.where(importance_small.feature.str.contains("fz"), "Fz",
								pd.np.where(importance_small.feature.str.contains("f3"), "F3",
								pd.np.where(importance_small.feature.str.contains("f4"), "F4",
								pd.np.where(importance_small.feature.str.contains("pz"), "Pz",
							 	pd.np.where(importance_small.feature.str.contains("p3"), "P3",
								pd.np.where(importance_small.feature.str.contains("f7"), "F7",
							 	pd.np.where(importance_small.feature.str.contains("f8"), "F8",
                   		     	pd.np.where(importance_small.feature.str.contains("cz"), "Cz",
								pd.np.where(importance_small.feature.str.contains("c3"), "C3",
                   		     	pd.np.where(importance_small.feature.str.contains("c4"), "C4",
							 	pd.np.where(importance_small.feature.str.contains("t3"), "T3",
							 	pd.np.where(importance_small.feature.str.contains("t4"), "T4",
							 	pd.np.where(importance_small.feature.str.contains("t5"), "T5",
								pd.np.where(importance_small.feature.str.contains("t6"), "T6",
								pd.np.where(importance_small.feature.str.contains("o1"), "O1",
							 	pd.np.where(importance_small.feature.str.contains("o2"), "O2", "miss")))))))))))))))))

importance_small = importance_small.sort_index()

"""CREATING VISUALIZATION FOR MOST IMPORTANCE FEATURES OVERALL
Creating bar charts to visualize the most important features in the model.
Custom colors are used to differentiate between frequency bands."""

delta_patch = mpatches.Patch(color='#5e01a6', label='Delta')
theta_patch = mpatches.Patch(color='#b7318a', label='Theta')
alpha_patch = mpatches.Patch(color='#ef7e50', label='Alpha')
beta_patch = mpatches.Patch(color='#fdc229', label='Beta')
pe_patch = mpatches.Patch(facecolor='white', edgecolor='black', label='PE')
dpli_patch = mpatches.Patch(facecolor='white', hatch="---", edgecolor='black', label='DPLI')

patterns = ["", "", "", "", "", "", "", "", "", "", "---", "", "", "", "", "---", "", "", ""]
electrodes_list = ['F4', 'C3', 'T5', 'Fz', 'F3', 'F4', 'C3', 'T3', 'Pz', '02', 'Fp2', 'Fz', 'F4', 'C4', 'O1', 'T5', 'C4', 'T3', "T4"]
colormap = ['#5e01a6', '#5e01a6', '#5e01a6', '#b7318a', '#b7318a', '#b7318a',
			   '#b7318a', '#b7318a', '#b7318a', '#b7318a', '#ef7e50', '#ef7e50',
			   '#ef7e50', '#ef7e50', '#ef7e50', '#fdc229', '#fdc229', '#fdc229', '#fdc229']

# using fill to differentiate between figure types
fig, ax = plt.subplots()
plt.suptitle('Baseline vs. CPT Condition: Feature Importance', fontweight='bold', fontsize=16)

# create bar chart for the PE features
ax.set_xticks(range(len(electrodes_list)))
ax.set_xticklabels(electrodes_list)
ax.bar(range(len(electrodes_list)), importance_small['model_coef'], color=colormap, hatch=patterns, edgecolor='black')
ax.axhline(y=0, color='black', linestyle='-')
ax.set_title('Chronic Pain Subjects', fontsize=14)
ax.axis(ymin=-0.45,ymax=0.70)
ax.set_xlabel("Electrodes", fontweight='bold', fontsize=12)
ax.set_ylabel("Model Weights (Logistic Regression)", fontweight='bold', fontsize=12)
ax.legend(handles=[delta_patch, theta_patch, alpha_patch, beta_patch, pe_patch, dpli_patch])

# DIFFERENT COLOR SCHEME

delta_patch = mpatches.Patch(color='#3e356b', label='Delta')
theta_patch = mpatches.Patch(color='#366ca0', label='Theta')
alpha_patch = mpatches.Patch(color='#3497a9', label='Alpha')
beta_patch = mpatches.Patch(color='#a9e1bd', label='Beta')
pe_patch = mpatches.Patch(facecolor='white', edgecolor='black', label='PE')
dpli_patch = mpatches.Patch(facecolor='white', hatch="---", edgecolor='black', label='DPLI')

patterns = ["", "", "", "", "", "", "", "", "", "", "---", "", "", "", "", "---", "", "", ""]
electrodes_list = ['F4', 'C3', 'T5', 'Fz', 'F3', 'F4', 'C3', 'T3', 'Pz', '02', 'Fp2', 'Fz', 'F4', 'C4', 'O1', 'T5', 'C4', 'T3', "T4"]
colormap = ['#3e356b', '#3e356b', '#3e356b', '#366ca0', '#366ca0', '#366ca0',
			   '#366ca0', '#366ca0', '#366ca0', '#366ca0', '#3497a9', '#3497a9',
			   '#3497a9', '#3497a9', '#3497a9', '#a9e1bd', '#a9e1bd', '#a9e1bd', '#a9e1bd']
