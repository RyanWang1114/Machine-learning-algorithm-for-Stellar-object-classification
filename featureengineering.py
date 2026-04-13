import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import math

import os

stellar = pd.read_csv('updated_stellar_dataset.csv')

import numpy as np
import pandas as pd

max_redshift = np.max(stellar["redshift"])
print(max_redshift)
min_redshift = np.min(stellar["redshift"])
print(min_redshift)

stellar["redshift"] = (stellar["redshift"] - min_redshift)/(max_redshift - min_redshift)

max_redshift = np.max(stellar["redshift"])
print(max_redshift)
min_redshift = np.min(stellar["redshift"])
print(min_redshift)

# Assuming you have already loaded your DataFrame named 'stellar'

def calculate_bounds(df, column):
    q25 = df[column].quantile(0.25)  # First quartile (25%)
    q75 = df[column].quantile(0.75)  # Third quartile (75%)
    iqr = q75 - q25                   # Interquartile range (IQR)
    lower_bound = q25 - 1.5 * iqr     # Lower bound
    upper_bound = q75 + 1.5 * iqr     # Upper bound
    return lower_bound, upper_bound

# Calculate bounds for each relevant column
bounds = {col: calculate_bounds(stellar, col) for col in ['u', 'g', 'r', 'i', 'z', 'redshift']}

# Print number of values within the IQR-adjusted range before filtering
for col in ['u', 'g', 'r', 'i', 'z', 'redshift']:
    count_before = stellar[(stellar[col] >= bounds[col][0]) & (stellar[col] <= bounds[col][1])].shape[0]
    print(f"Number of {col} values within the IQR-adjusted range before filtering: {count_before}")


# Filter the DataFrame by keeping only rows where all specified columns are within their IQR bounds
filtered_stellar = stellar[
    (stellar['u'] >= bounds['u'][0]) & (stellar['u'] <= bounds['u'][1]) &
    (stellar['g'] >= bounds['g'][0]) & (stellar['g'] <= bounds['g'][1]) &
    (stellar['r'] >= bounds['r'][0]) & (stellar['r'] <= bounds['r'][1]) &
    (stellar['i'] >= bounds['i'][0]) & (stellar['i'] <= bounds['i'][1]) &
    (stellar['z'] >= bounds['z'][0]) & (stellar['z'] <= bounds['z'][1]) &
    (stellar['redshift'] >= bounds['redshift'][0]) & (stellar['redshift'] <= bounds['redshift'][1])
]

# Print number of values within the IQR-adjusted range after filtering
for col in ['u', 'g', 'r', 'i', 'z', 'redshift']:
    count_after = filtered_stellar[(filtered_stellar[col] >= bounds[col][0]) & (filtered_stellar[col] <= bounds[col][1])].shape[0]
    print(f"Number of {col} values within the IQR-adjusted range after filtering: {count_after}")

# Optionally print the filtered DataFrame
print("\nFiltered DataFrame:")
print(filtered_stellar)










# Create a histogram for the "alpha" column
#stellar["u"].hist()
#plt.xlabel('u')
#plt.ylabel('sample')
#plt.title('u')
#plt.show()







print(" ")
print(" ")

q75, q25 = np.percentile(stellar["alpha"], [75,25])
print("alpha")
iqr = q75 - q25
print(iqr, q75, q25)

stellar["alpha_cut"] = pd.cut(stellar["alpha"], bins = 12, labels=False)

q75, q25 = np.percentile(stellar["delta"], [75,25])
print("delta")
iqr = q75 - q25
print(iqr, q75, q25)

stellar["delta_cut"] = pd.cut(stellar["delta"], bins = 12, labels = False)

print(" ")
print(" ")
print(" ")
print(" ")
print(" ")
print(" ")




stellar["distance_to_nearest_neighbor"] = np.sqrt(stellar["alpha"]**2 + stellar["delta"]**2)
print("")
stellar["u-g"] = stellar["u"]-stellar["g"]
stellar["g-r"] = stellar["g"]-stellar["r"]
stellar["r-i"] = stellar["r"]-stellar["i"]
stellar["i-z"] = stellar["i"]-stellar["z"]


print(stellar.head())

stellar.to_csv('ready to go.csv', index=False)
print("Updated dataset has been saved to 'updated_stellar_dataset.csv'.")



