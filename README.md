# Higgs-project
code for StoM higgs project
"""
Created on Fri May  9 15:17:38 2025

@author: josephgolden
"""
# task 1 start:
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
#%%
np.random.seed(1)

N_b = 10e5 # Number of background events, used in generation and in fit.
b_tau = 30. # Spoiler.

def generate_signal(N, mu, sig):
    ''' 
    Generate N values according to a gaussian distribution.
    '''
    return np.random.normal(loc = 0, scale = 1, size = 400).tolist()

# %%

def generate_background(N, tau):
    ''' 
    Generate N values according to an exp distribution.
    '''
 
    return np.random.exponential(scale = tau, size = int(N)).tolist()
def generate_data(n_signals = 400):
    

    ''' 
    Generate a set of values for signal and background. Input arguement sets 
    the number of signal events, and can be varied (default to higgs-like at 
    announcement). 
    
    The background amplitude is fixed to 9e5 events, and is modelled as an exponential, 
    hard coded width. The signal is modelled as a gaussian on top (again, hard 
    coded width and mu).
    '''
    vals = []
    vals += generate_signal( n_signals, 125., 1.5)
    vals += generate_background( N_b, b_tau)
    
    return vals

vals = generate_data()

# Make a histogram.
bin_heights, bin_edges, patches = plt.hist(vals, range = [104, 155], bins = 30)
params = {
   'axes.labelsize': 8,
   'font.size': 8,
   'legend.fontsize': 8,
   'xtick.labelsize': 8,
   'ytick.labelsize': 8,
   'figure.figsize': [5,8.09] #using golden ratio
   } 
bins_c1 = plt.hist(vals, range = [50, 200], bins = 30, color='red', label = '50-200')

bins_main = plt.hist(vals, range = [104, 155], bins = 30, color='green' ,label = '104-155')

bins_c2 = plt.hist(vals, range = [120, 150], bins = 30, color='blue', label = '120-150')
bincentres = 0.5 * (bins_main[1][1:] + bins_main[1][:-1])  
# calculate the bin centres (i.e. x position of the error bars)
print('x,y coordinates of bincentres are', bincentres)
print('y coordinates of bincentres are', bins_main[0])


def fit_func():
    return generate_background(400, 30) + generate_signal(400, 125, 1.5)
#%%


params = {
   'axes.labelsize': 8,
   'font.size': 8,
   'legend.fontsize': 8,
   'xtick.labelsize': 8,
   'ytick.labelsize': 8,
   'figure.figsize': [5,8.09] #using golden ratio
   } 
plt.errorbar(bincentres, bins_c1[0], yerr = np.sqrt(bins_c1[0]), color='red',
             fmt='o', mew=2, ms=3, capsize=4)  # plot the error bars.

plt.errorbar(bincentres, bins_main[0], yerr = np.sqrt(bins_main[0]), color='green',
             fmt='o', mew=2, ms=3, capsize=4)  # plot the error bars.

plt.errorbar(bincentres, bins_c2[0], yerr = np.sqrt(bins_c2[0]), color='blue',
             fmt='o', mew=2, ms=3, capsize=4)  # plot the error bars.

plt.xlabel(r'$m_{\gamma\gamma}$ (GeV)', loc = 'right')
plt.ylabel("Number of entries")
plt.legend(['Data points - range: 50-200', 'Data points - range: 104-155', 'Data points - range: 120-150'],loc='upper right', fontsize=12, frameon=False)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.tight_layout()
plt.show()
# bin_heights and bin_edges are numpy arrays.
# patches are the matplotlib bar objects, which we won’t need.
# For larger bin sizes, signal disappears
# %%
plt.errorbar(bincentres, bins_main[0], yerr = np.sqrt(bins_main[0]), color='black',
             fmt='o', mew=2, ms=3, capsize=4)  # plot the error bars.
plt.xlabel(r'$m_{\gamma\gamma}$ (GeV)', loc = 'right')
plt.ylabel("Number of entries")
plt.legend(['Data points',],loc='upper right', fontsize=12, frameon=False)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.tight_layout()
plt.show()
#task one end:
#Task 3 starts
import numpy as np
import matplotlib.pyplot as plt

# The function is defined to generate background expectation values
def get_B_expectation(xs, A, lamb):
    return A * np.exp(-xs / lamb)

# The function is used to generate chi square function 
def get_B_chi(vals, histo_range, n_bins, A, lamb):
    bin_heights, bin_edges = np.histogram(vals, range=histo_range, bins=n_bins)
    bin_centres = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    expected = get_B_expectation(bin_centres, A, lamb)
    
    errors = np.sqrt(bin_heights)
    errors[errors == 0] = 1e-6 # in case division by 0 in the functions

    chi2 = np.sum(((bin_heights - expected) ** 2) / errors ** 2)
    return chi2

# Parameters for the histogram
range_low, range_up = 104, 155
n_bins = 30
bin_centres = np.linspace(range_low + (range_up - range_low) / (2 * n_bins),
                          range_up - (range_up - range_low) / (2 * n_bins), n_bins)

vals = generate_data()

# Choose scan ranges for A and Lambda
A_vals = np.linspace(1e3, 1e5, 100)# Try values from 1000 to 100000
lamb_vals = np.linspace(20, 40, 100)# Try values from 20 to 40

chi2_grid = np.zeros((len(A_vals), len(lamb_vals)))

# For each combination, calculate the χ² and store it.
for i, A in enumerate(A_vals):
    for j, lamb in enumerate(lamb_vals):
        chi2 = get_B_chi(vals, (range_low, range_up), n_bins, A, lamb)
        chi2_grid[i, j] = chi2

# Find where the minimum χ² occurs
min_index = np.unravel_index(np.argmin(chi2_grid), chi2_grid.shape)
best_A = A_vals[min_index[0]]
best_lamb = lamb_vals[min_index[1]]
min_chi2 = chi2_grid[min_index]

print(f"Best-fit A: {best_A}")
print(f"Best-fit lambda: {best_lamb}")
print(f"Minimum chi²: {min_chi2}")

# Plot background fit on histogram
bin_heights, bin_edges, _ = plt.hist(vals, range=(range_low, range_up), bins=n_bins, label='Data', alpha=0.5)
expected = get_B_expectation(bin_centres, best_A, best_lamb)
plt.plot(bin_centres, expected, label='Min-χ² Fit', color='red')
plt.xlabel(r'$m_{\gamma\gamma}$ (GeV)')
plt.ylabel('Entries')
plt.legend()
plt.tight_layout()
plt.show()
#task 3 ends
