from scipy.stats import norm, expon
#higgs tools: raw data
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)

N_b = 10e5 # Number of background events, used in generation and in fit.
b_tau = 30. # Spoiler.

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


def generate_signal(N, mu, sig):
    ''' 
    Generate N values according to a gaussian distribution.
    '''
    return np.random.normal(loc = mu, scale = sig, size = N).tolist()


def generate_background(N, tau):
    ''' 
    Generate N values according to an exp distribution.
    '''
    return np.random.exponential(scale = tau, size = int(N)).tolist()

# task 2:
vals = generate_data()

def em(data, max_iter=1000, tol=1e-6):
    alpha = 0.5  # initial values
    mu = np.mean(data)  
    sigma = np.std(data)  
    lamb = 1 / np.mean(data)  
    
    prev_params = np.array([alpha, mu, sigma, lamb])
    
    for _ in range(max_iter):
        # weight of normal
        prob_normal = alpha * norm.pdf(data, mu, sigma)
        prob_exp = (1 - alpha) * expon.pdf(data, scale=1/lamb)
        weights = prob_normal / (prob_normal + prob_exp + 1e-10)  # avoid dividing by 0
        
        # update
        alpha_new = np.mean(weights)
        mu_new = np.sum(weights * data) / (np.sum(weights) + 1e-10)
        sigma_new = np.sqrt(np.sum(weights * (data - mu_new)**2) / (np.sum(weights) + 1e-10))
        lamb_new = np.sum(1 - weights) / (np.sum((1 - weights) * data) + 1e-10)
        
        # convergence check
        new_params = np.array([alpha_new, mu_new, sigma_new, lamb_new])
        if np.max(np.abs(new_params - prev_params)) < tol:
            break
        prev_params = new_params
    
    return {
        "alpha": alpha_new,
        "mu": mu_new,
        "sigma": sigma_new,
        "lambda": lamb_new
    }

result = em(vals)
print(f"lambda= {result['lambda']:.4f}")

# curve fitting
plt.figure(figsize=(10, 6))
n, bins, _ = plt.hist(vals, bins=40, density=True, alpha=0.5, label='Data')

x = np.linspace(0, np.max(vals), 1000)
pdf_normal = result['alpha'] * norm.pdf(x, result['mu'], result['sigma'])
pdf_exp = (1 - result['alpha']) * expon.pdf(x, scale=1/result['lambda'])
plt.plot(x, pdf_normal + pdf_exp, 'r-', lw=2, label='Fitted Mixture')
plt.plot(x, pdf_exp, 'g--', lw=1.5, label='Exponential Component')
plt.legend()
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Fitted PDF')
plt.show()