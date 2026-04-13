# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Loading the dataset
stellar = pd.read_csv('star_classification (2).csv')

# Displaying the first few rows of the dataset
stellar.head()

# Showing information about the dataset, including column types and null counts
stellar.info()

# Displaying summary statistics for each column in the dataset
# pd.set_option('display.max_columns', 18)  # Uncomment to adjust display settings for more columns
# pd.set_option('display.max_rows', 11)     # Uncomment to adjust display settings for more rows
stellar.describe(include='all')

# Counting unique values in 'obj_ID' and 'spec_obj_ID' columns to check for duplication
stellar['obj_ID'].nunique(), stellar['spec_obj_ID'].nunique()
print(f"Unique obj_ID count: {stellar['obj_ID'].nunique()}, Unique spec_obj_ID count: {stellar['spec_obj_ID'].nunique()}")



# Converting 'class' column to numerical values: 0 for GALAXY, 1 for STAR, and 2 otherwise
stellar["class"] = [0 if i == "GALAXY" else 1 if i == "STAR" else 2 for i in stellar["class"]]

# # Optional: Plotting heatmap for correlation analysis between features
f, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(stellar.corr(), annot=True, linewidths=0.5, fmt='.2f', ax=ax)
plt.show()

# # Optional: Displaying correlations with 'class' column sorted for reference
print(pd.DataFrame(stellar.corr()['class']).sort_values(by=['class'], ascending=True))



# Displaying unique values in the 'class' column to confirm encoding
stellar["class"].unique()

# Checking if 'alpha' values are within 0-360 degrees and counting valid entries
alphacount = ((stellar['alpha'] >= 0) & (stellar['alpha'] <= 360)).sum()
print(f"alpha = {alphacount}")

# Checking if 'delta' values are within -90 to 90 degrees and counting valid entries
deltacount = ((stellar['delta'] >= -90) & (stellar['delta'] <= 90)).sum()
print(f"delta = {deltacount}")

# Defining list of columns with specific value ranges for further inspection
columns = ['u', 'g', 'r', 'i', 'z']

# Checking if each column in 'columns' has values within -10 to 30 and counting valid entries
for col in columns:
    count = ((stellar[col] >= -10) & (stellar[col] <= 30)).sum()
    print(f"{col} = {count}")

# Checking if 'cam_col' values are within the range 1-6 and counting valid entries
count = ((stellar['cam_col'] >= 1) & (stellar['cam_col'] <= 6)).sum()
print(f"cam_col = {count}")

# Checking if 'redshift' values are within the range 0-7 and counting valid entries
count = ((stellar['redshift'] >= 0) & (stellar['redshift'] <= 7)).sum()
print(f"redshift = {count}")

# Checking if 'MJD' values are within 50000-60000 and counting valid entries
count = ((stellar['MJD'] >= 50000) & (stellar['MJD'] <= 60000)).sum()
print(f"MDJ = {count}")





# Calculating the mean of 'redshift' within the range 0-7 for imputation purposes
mean_redshift = stellar[(stellar['redshift'] >= 0) & (stellar['redshift'] <= 7)]['redshift'].mean()

# Print the mean value of 'redshift' within range 0 to 7
print(f"Mean of 'redshift' within range 0 to 7: {mean_redshift}")

# Step 2: Replace 'redshift' values outside 0-7 with the calculated mean
stellar.loc[(stellar['redshift'] < 0) | (stellar['redshift'] > 7), 'redshift'] = mean_redshift

# Defining columns to process for replacing out-of-range values
columns_to_process = ['u', 'g', 'i', 'z']

for col in columns_to_process:
    # Calculate the mean for values within -10 to 30
    mean_value = stellar[(stellar[col] >= -10) & (stellar[col] <= 30)][col].mean()
    
    # Print the mean value for reference
    print(f"Mean of '{col}' within range -10 to 30: {mean_value}")

    # Replace values not within the range -10 to 30 with the calculated mean
    stellar.loc[(stellar[col] < -10) | (stellar[col] > 30), col] = mean_value








    

# Re-checking counts after replacements to ensure integrity of 'alpha' and 'delta' columns
alphacount = ((stellar['alpha'] >= 0) & (stellar['alpha'] <= 360)).sum()
print(f"alpha = {alphacount}")

deltacount = ((stellar['delta'] >= -90) & (stellar['delta'] <= 90)).sum()
print(f"delta = {deltacount}")

# Re-checking counts in 'u', 'g', 'r', 'i', 'z' columns to confirm replacement accuracy
for col in columns:
    count = ((stellar[col] >= -10) & (stellar[col] <= 30)).sum()
    print(f"{col} = {count}")

# Re-checking counts in 'cam_col', 'redshift', and 'MJD' columns to confirm replacement accuracy
count = ((stellar['cam_col'] >= 1) & (stellar['cam_col'] <= 6)).sum()
print(f"cam_col = {count}")

count = ((stellar['redshift'] >= 0) & (stellar['redshift'] <= 7)).sum()
print(f"redshift = {count}")

count = ((stellar['MJD'] >= 50000) & (stellar['MJD'] <= 60000)).sum()
print(f"MDJ = {count}")


# # Optional: Checking for null values and unique counts in the dataset
# print(stellar.isnull().sum())
# print(stellar.nunique())

# # Filtering rows based on value ranges for 'u', 'g', 'r', 'i', 'z' columns if needed
# stellar = stellar[(stellar[columns] >= -10).all(axis=1) & (stellar[columns] <= 30).all(axis=1)]

# Saving the updated dataset to a new CSV file
# stellar.to_csv('updated_stellar_dataset.csv', index=False)

# Optional: Print a message to confirm the export
print("Updated dataset has been saved to 'updated_stellar_dataset.csv'.")
