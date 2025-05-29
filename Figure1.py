import numpy as np
import os,gc
import matplotlib.pyplot as plt
from Objective_functions import objective_2log

def objective(sol):
    return objective_2log(sol)+10

def E_fx(x,y,mu,sigma=1.0):
    """
    This function computes F(mu):=E[f(mu+sigma*xi)], where xi is 
    a standard normal random variable, and y=f(x)
    
    Inputs.
    -------
    x:np.1darray
    y:np.1darray, y=f(x)
    mu:float
    sigma:float
    
    Outputs.
    --------
    integral:float, integral = E[f(mu+sigma*xi)]
    """
    x_delta = x[1]-x[0]
    coeff = 1/(np.sqrt(2*np.pi)*sigma) 
    integral = coeff * x_delta * np.sum(   np.exp(-(x-mu)**2/(2*sigma**2)) * y   ) 
    return integral

############### - E(f^N(x)) 
#Compare to the above, the maximum of the Guassian smoothing is 
#closer to that of the original obj.
x_delta = 0.01
x_min = -1
x_max = 1
x = np.arange(x_min,x_max,x_delta)
y = objective(x.reshape([-1,1]))       #f(x)

#Compute the integral
mu_delta = 0.01
mu_min = -1.0
mu_max = 1.0
sigma = 0.5
mu_vec = np.arange(mu_min,mu_max,x_delta)


plt.figure(figsize=(3.2,4.5))
#plt.rcParams['text.usetex'] = True
plt.plot(x,y/max(y),label=r"$f(\mu)$",color="red")
F_mu = [E_fx(x,y,mu,sigma) for mu in mu_vec]
plt.plot(mu_vec,F_mu/max(F_mu),label=r'$N=1$', linestyle="dotted",color="purple")
F_mu = [E_fx(x,y**5,mu,sigma) for mu in mu_vec]
plt.plot(mu_vec, F_mu/max(F_mu),label=r'$N=5$', 
         linestyle="-.",color="green")
F_mu = [E_fx(x,y**15,mu,sigma) for mu in mu_vec]
plt.plot(mu_vec, F_mu/max(F_mu),label=r'$N=15$', 
         linestyle="--",color="blue")
plt.legend(loc="lower left",fontsize=16)
#plt.title("Power Transform",size=20)
plt.xlabel("$\mu$",size=20)
plt.tick_params(labelsize=15)
plt.savefig("Fig1-a.eps",bbox_inches='tight', transparent=True) #
plt.show()


############### - E[exp(Nf(x))] 
#Compare to the above, the maximum of the Guassian smoothing is 
#closer to that of the original obj.
x_delta = 0.01
x_min = -1
x_max = 1
x = np.arange(x_min,x_max,x_delta)
y = objective(x.reshape([-1,1]))       #f(x)

#Compute the integral
mu_delta = 0.01
mu_min = -1.0
mu_max = 1.0
sigma = 0.5
mu_vec = np.arange(mu_min,mu_max,x_delta)


plt.figure(figsize=(3.2,4.5))
#plt.rcParams['text.usetex'] = True
plt.plot(x,y/max(y),label=r"$f(\mu)$",color="red")
F_mu = [E_fx(x,np.exp(0.1*y),mu,sigma) for mu in mu_vec]
plt.plot(mu_vec,F_mu/max(F_mu),label=r'$N=0.1$', linestyle="dotted",color="purple")
F_mu = [E_fx(x,np.exp(0.4*y),mu,sigma) for mu in mu_vec]
plt.plot(mu_vec, F_mu/max(F_mu),label=r'$N=0.4$', 
         linestyle="-.",color="green")
F_mu = [E_fx(x,np.exp(y),mu,sigma) for mu in mu_vec]
plt.plot(mu_vec, F_mu/max(F_mu),label=r'$N=1$', 
         linestyle="--",color="blue")
#plt.title("Exp. Transform",size=20)
plt.legend(loc="lower left",fontsize=16)
plt.xlabel("$\mu$",size=20)
plt.tick_params(labelsize=15)
plt.savefig("Fig1-b.eps",bbox_inches='tight', transparent=True)#
plt.show()