import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma as g

# Regular NGa - pdf, used to describe the exact posterior for mu and tau
def normalGamma(m, l, a, b):
    C = (b**a)*np.sqrt(l) / (g(a) * np.sqrt(2*np.pi))
    return lambda x, tau:C*tau**(a-0.5)*np.exp(-b*tau-0.5*l*tau*(x-m)**2)

# Precision prior
def gamma(a,b):
    C = b**a/g(a)
    return lambda tau: C * tau**(a-1)*np.exp(-b*tau)

# Mean prior
def normal(mu,sigma):
    C = np.sqrt(2*np.pi*sigma**2)
    return lambda m: C * np.exp(-(m-mu)**2*0.5/sigma**2)

# VI approximated posterior
def vi(a,b,mu,l):
    return lambda x, tau: normal(mu,1./np.sqrt(l))(x)*gamma(a,b)(tau)

np.random.seed(1)
N = 5

# Generate Gaussian univariate samples
X  = np.random.normal(0, 1, N) 
m  = np.mean(X)
v  = np.var(X)

# Initialize prior
a_0 = 1; b_0 = 1; l_0 = 1; mu_0 = 0

# Calculate true posterior
a_t  = a_0 + N/2
l_t  = l_0 + N 
mu_t = (l_0*mu_0 + N*m) / (l_0 + N)
b_t  = b_0 + 1./2*(N*v + (l_0*N*(m - mu_0)**2)/(l_0 + N))
x = np.linspace(-2, 2, 100)
y = np.linspace(0, 5, 100)
Z1 = normalGamma(mu_t, l_t, a_t, b_t)(*np.meshgrid(x, y))

# VI - parameters
mu_v = (l_0*mu_0 + N*m)/(l_0 + N)
a_v  = a_0 + (N+1)/2.

# Initial guess
l_v = 1

# 100 iterations which strongly suggested convergence after some testing
for _ in range(100): 
        b_v  = b_0 + 0.5 * ((l_0 + N) * (1./l_v + mu_v**2)
                            - 2 * (l_0 * mu_0 + N*m) * mu_v 
                            + np.sum(X**2) + l_0*mu_0**2)        
        l_v  = (l_0 + N)*(a_v/b_v)
        
# Plotting
fig, ax = plt.subplots()
Z2 = vi(a_v, b_v, mu_v, l_v)(*np.meshgrid(x, y))        
cntr1 = ax.contour(x, y, Z1, colors='k')
cntr2 = ax.contour(x, y, Z2, colors='r')
h1,_ = cntr1.legend_elements()
h2,_ = cntr2.legend_elements()
ax.legend([h1[0], h2[0]], ['True', 'Approximation'])
ax.set_xlabel('$\mu$', fontsize=15)
ax.set_ylabel('$\\tau$', fontsize=15)
plt.savefig('Case3a.eps', bbox_inches='tight')
plt.show()