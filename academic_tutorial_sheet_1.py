# ACADEMIC TUTORIAL SHEET 1 - MY CODE

# imports

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as st

# (1) construct a suitable histogram plot for the geiger counter data

geiger_counter_data = np.array([15, 25, 22, 31, 25, 19, 8, 24, 44, 30, 34, 12, 7, 33, 19, 20, 19, 42, 38, 27])

geiger_histogram_bins = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45])
geiger_histogram = plt.hist(geiger_counter_data, bins = geiger_histogram_bins, edgecolor = 'black')
plt.xlabel('Number of Alpha Particles in 1 Minute')
plt.ylabel('Frequency')
plt.show()

# (2) determine the sample mean, the sample variance, the median and the mode of the geiger counter data

mean = np.mean(geiger_counter_data)
variance = np.var(geiger_counter_data)
median = np.median(geiger_counter_data)
mode = np.bincount(geiger_counter_data).argmax()

print(mean, variance, median, mode)