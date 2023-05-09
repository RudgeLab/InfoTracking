import numpy as np
from skimage.io import imread, imsave
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

# Compute average growth rate of colony at each time point
area = np.load('area.npy')
sarea = savgol_filter(area, 11, 3)
dsarea = savgol_filter(area, 11, 3, deriv=1)
mean_growth_rate = dsarea / area
np.save('mean_growth_rate.npy', mean_growth_rate)

# Compute colony edge velocity
radius = np.load('radius.npy')
vfront = savgol_filter(radius, 11, 3, deriv=1)
np.save('vfront.npy', vfront)


