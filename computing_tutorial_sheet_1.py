# COMPUTING TUTORIAL SHEET 1 - MY CODE

# imports

import numpy as np
import pandas as pd
from scipy import stats as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# (1) importing data and summary statistics -----------------------------------------------------------------------------------------------------------------------------------

faithful = pd.read_csv('faithful.csv')

# print(faithful.head())

waiting = faithful['waiting']

# print("Mean:", np.mean(waiting))
# print("Median:", np.median(waiting))
# print("Mode:", np.bincount(waiting).argmax())
# print("Standard deviation:", np.std(waiting, ddof=1))  # sample std

plt.figure(figsize=(8,5))
sns.boxplot(x = 'day', y = 'waiting', data = faithful, color = 'yellow')
plt.xlabel("Day")
plt.ylabel("Waiting time between successive eruptions (mins)")
# plt.show()

# (2) histograms and kernel plots ---------------------------------------------------------------------------------------------------------------------------------------------

min = np.min(waiting)
max = np.max(waiting)

n_bins = 50 # try 9, 50

bin_edges = np.linspace(np.min(waiting), np.max(waiting), n_bins + 1)
counts, _ = np.histogram(waiting, bins = bin_edges)

rel_freq = counts / len(waiting)

plt.figure(figsize = (8,5))
sns.histplot(waiting, bins = n_bins, color = 'blue', stat = 'density', kde_kws = {'color': 'red'})
sns.kdeplot(waiting, color='red', linewidth=2) 
plt.xlabel("Waiting time (mins)")
plt.ylabel("Relative frequency")
plt.title("Histogram of waiting")
# plt.show()

sns.kdeplot(waiting, color='red', linewidth=2, label='bw = default')
sns.kdeplot(waiting, bw_adjust=2, color='green', linewidth=1, label='bw = 2')
sns.kdeplot(waiting, bw_adjust=0.75, color='blue', linewidth=1, label='bw = 0.75')
plt.xlabel("Waiting time (mins)")
plt.ylabel("Density")
plt.legend()
# plt.show()

# (3) plotting consecutive eruption waiting times ----------------------------------------------------------------------------------------------------------------------------

days = faithful['day'].unique()
days.sort()

fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(15,10))
axes = axes.flatten()

for i, day in enumerate(days):
    day_data = faithful.loc[faithful['day'] == day, 'waiting']
    axes[i].plot(range(1, len(day_data)+1), day_data, linestyle='-')
    axes[i].set_title(f"Day: {day}")
    axes[i].set_ylabel("Waiting time (mins)")
    axes[i].set_ylim((day_data.min() - 10), (day_data.max() + 10))

plt.tight_layout()
plt.show()

# (4) scatterplots and linear regression ------------------------------------------------------------------------------------------------------------------------------------

n = len(faithful) # found this to be 285

lagduration = faithful['duration'].shift(1)

waiting = faithful['waiting'][1:]
lagduration = lagduration[1:]

B = np.polyfit(lagduration, waiting, 1)
print(f"waitingest = {B[1]:.3f} + {B[0]:.3f} * lagduration")

waitingest = B[0] * lagduration + B[1]

plt.figure(figsize = (8,5))
plt.scatter(lagduration, waiting, color = 'orange')
plt.plot(lagduration, waitingest, color = 'blue', linewidth = 1)
plt.xlabel("Previous duration (minutes)")
plt.ylabel("Waiting time (minutes)")
plt.show()

# (5) k-means clustering ----------------------------------------------------------------------------------------------------------------------------------------------------

X = pd.DataFrame({'lagduration': lagduration, 'waiting': waiting})

K = 2
C = KMeans(n_clusters = K, random_state = 0).fit(X)

X['cluster'] = C.labels_

label0 = X.loc[C.labels_==0] 
label1 = X.loc[C.labels_==1] 
    
plt.scatter(x=label0['lagduration'], y=label0['waiting'], color = 'red')
plt.scatter(x=label1['lagduration'], y=label1['waiting'], color = 'green')
plt.show()

# (6) final questions ------------------------------------------------------------------------------------------------------------------------------------------------------

