from sklearn.metrics import mean_squared_error as mse
import numpy as np
import pandas as pd
import os,gc
from Smoothing_Methods import power_gs, power_gs_baseline, exp_gs_adapt, evaluate_gs
from Objective_functions import objective_2log, plot_obj_2log, plot_Ackley, plot_Rosenbrock
import matplotlib.pyplot as plt

#Plot the objective - Figure 3(a),(b),(c)
plot_obj_2log(savefig=False)
plot_Ackley(savefig=False)
plot_Rosenbrock(savefig=False)

#exp_gs 2D
N_list_epgs_2d = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5] #
pop_size = 100
state_dim = 2
generation_num = 500
true_sol = np.array([-0.5]*state_dim)
fitness_list_epgs_2d = []
mse_list_epgs_2d = []
best_sol_time_list = []
repeat_num = 100 #the number of experiments repeated for each value of N
r = np.random.RandomState(0)
for N in N_list_epgs_2d:
    print("N",N)
    mse_holder = []
    fit_holder = []
    for i in range(repeat_num):
        init_mu = r.uniform(low=-1.0, high=1.0, size=state_dim)
        mu_log = exp_gs_adapt(
                                        init_mu=init_mu, 
                                        dim=state_dim, 
                                        fitness_fcn=objective_2log,
                                        init_lr=0.1, 
                                        sigma=1.0,
                                        power=N,
                                        sga_sample_size=pop_size,
                                        total_step_limit=generation_num, 
                        )
        best_mu_time, best_mu, best_mu_fit, best_mse = evaluate_gs(mu_log, objective_2log, true_sol, verbose=False)
        mse_holder.append(best_mse)
        fit_holder.append(best_mu_fit)
    avg_fit = np.mean(fit_holder)
    avg_mse = np.mean(mse_holder)
    print("Best avg fit:",round(avg_fit,3))
    print("avg mse(global_max,best_fit):",round(avg_mse,10))
    fitness_list_epgs_2d.append(avg_fit)
    mse_list_epgs_2d.append(avg_mse)
    #best_sol_time_list.append(best_mu_time)
    print(" ")

res = pd.DataFrame({"N":N_list_epgs_2d, "Avg Fitness":fitness_list_epgs_2d, "Avg mse":mse_list_epgs_2d})
plt.figure()
plt.plot(N_list_epgs_2d, fitness_list_epgs_2d)
plt.show()

#power_gs 2D
N_list_pgs_2d = [10, 20, 30, 35, 40, 45, 50, 55]
pop_size = 100
state_dim = 2
generation_num = 500
true_sol = np.array([-0.5]*state_dim)
fitness_list_pgs_2d = []
mse_list_pgs_2d = []
best_sol_time_list = []
repeat_num = 100 #the number of experiments repeated for each value of N
def power_objective(sol):
    return objective_2log(sol)+10
r = np.random.RandomState(0)
for N in N_list_pgs_2d:
    print("N",N)
    mse_holder = []
    fit_holder = []
    for i in range(repeat_num):
        init_mu = r.uniform(low=-1.0, high=1.0, size=state_dim)
        mu_log = power_gs_baseline(
                                        init_mu=init_mu, 
                                        dim=state_dim, 
                                        fitness_fcn=power_objective,
                                        init_lr=0.1, 
                                        sigma=1.0,
                                        power=N,
                                        sga_sample_size=pop_size,
                                        total_step_limit=generation_num, 
                        )
        best_mu_time, best_mu, best_mu_fit, best_mse = evaluate_gs(mu_log, objective_2log, true_sol, verbose=False)
        mse_holder.append(best_mse)
        fit_holder.append(best_mu_fit)
    avg_fit = np.mean(fit_holder)
    avg_mse = np.mean(mse_holder)
    print("Best avg fit:",round(avg_fit,3))
    print("avg mse(global_max,best_fit):",round(avg_mse,10))
    fitness_list_pgs_2d.append(avg_fit)
    mse_list_pgs_2d.append(avg_mse)
    #best_sol_time_list.append(best_mu_time)
    print(" ")

res = pd.DataFrame({"N":N_list_pgs_2d, "Avg Fitness":fitness_list_pgs_2d, "Avg mse":mse_list_pgs_2d})
plt.figure()
plt.plot(N_list_pgs_2d, fitness_list_pgs_2d)
plt.show()


#exp_gs 5D
N_list_epgs_5d = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5] #
pop_size = 2000
state_dim = 5
generation_num = 1000
true_sol = np.array([-0.5]*state_dim)
fitness_list_epgs_5d = []
mse_list_epgs_5d = []
best_sol_time_list = []
repeat_num = 100 #the number of experiments repeated for each value of N
r = np.random.RandomState(0)
for N in N_list_epgs_5d:
    print("N",N)
    mse_holder = []
    fit_holder = []
    for i in range(repeat_num):
        init_mu = r.uniform(low=-1.0, high=1.0, size=state_dim)
        mu_log = exp_gs_adapt(
                                        init_mu=init_mu, 
                                        dim=state_dim, 
                                        fitness_fcn=objective_2log,
                                        init_lr=0.1, 
                                        sigma=1.0,
                                        power=N,
                                        sga_sample_size=pop_size,
                                        total_step_limit=generation_num, 
                        )
        best_mu_time, best_mu, best_mu_fit, best_mse = evaluate_gs(mu_log, objective_2log, true_sol, verbose=False)
        mse_holder.append(best_mse)
        fit_holder.append(best_mu_fit)
    avg_fit = np.mean(fit_holder)
    avg_mse = np.mean(mse_holder)
    print("Best avg fit:",round(avg_fit,3))
    print("avg mse(global_max,best_fit):",round(avg_mse,4))
    fitness_list_epgs_5d.append(avg_fit)
    mse_list_epgs_5d.append(avg_mse)
    #best_sol_time_list.append(best_mu_time)
    print(" ")

res = pd.DataFrame({"N":N_list_epgs_5d, "Avg Fitness":fitness_list_epgs_5d, "Avg mse":mse_list_epgs_5d})
plt.figure()
plt.plot(N_list_epgs_5d, fitness_list_epgs_5d)
plt.show()

#power_gs 5D
N_list_pgs_5d = [10, 20, 30, 35, 40, 45, 50, 55]
pop_size = 1000
state_dim = 5
generation_num = 1000
true_sol = np.array([-0.5]*state_dim)
fitness_list_pgs_5d = []
mse_list_pgs_5d = []
best_sol_time_list = []
repeat_num = 100 #the number of experiments repeated for each value of N
def power_objective(sol):
    return objective_2log(sol)+10
r = np.random.RandomState(0)
for N in N_list_pgs_5d:
    print("N",N)
    mse_holder = []
    fit_holder = []
    for i in range(repeat_num):
        init_mu = r.uniform(low=-1.0, high=1.0, size=state_dim)
        mu_log = power_gs_baseline(
                                        init_mu=init_mu, 
                                        dim=state_dim, 
                                        fitness_fcn=power_objective,
                                        init_lr=0.1, 
                                        sigma=1.0,
                                        power=N,
                                        sga_sample_size=pop_size,
                                        total_step_limit=generation_num, 
                        )
        best_mu_time, best_mu, best_mu_fit, best_mse = evaluate_gs(mu_log, objective_2log, true_sol, verbose=False)
        mse_holder.append(best_mse)
        fit_holder.append(best_mu_fit)
    avg_fit = np.mean(fit_holder)
    avg_mse = np.mean(mse_holder)
    print("Best avg fit:",round(avg_fit,3))
    print("avg mse(global_max,best_fit):",round(avg_mse,4))
    fitness_list_pgs_5d.append(avg_fit)
    mse_list_pgs_5d.append(avg_mse)
    #best_sol_time_list.append(best_mu_time)
    print(" ")

res = pd.DataFrame({"N":N_list_pgs_5d, "Avg Fitness":fitness_list_pgs_5d, "Avg mse":mse_list_pgs_5d})
plt.figure()
plt.plot(N_list_pgs_5d, fitness_list_pgs_5d)
plt.show()


#Plot
#epgs
plt.figure(figsize=(6,5))
plt.rcParams['text.usetex'] = True
plt.title("$f(\mu)$ - EPGS",size=12)
plt.plot(N_list_epgs_2d,fitness_list_epgs_2d,label=r'2d case epgs', linestyle="--",color="blue")
plt.plot(N_list_epgs_5d,fitness_list_epgs_5d,label=r'5d case epgs', linestyle="-",color="red")
plt.legend(loc="lower right",fontsize=14)
plt.xlabel("$\mu$",size=20)
plt.tick_params(labelsize=15)
fig_path = "C:/Users/HP/Desktop/Optimization with Objective Trans and Guassian Smoothing/Drafts/Figures/"
#plt.savefig(fig_path+"Fig2-epgs-fit.eps",bbox_inches='tight', transparent=True) #
plt.show()

plt.figure(figsize=(6,5))
plt.rcParams['text.usetex'] = True
plt.title(r"$mse(\mathbf{x}^*,\mu),$ - EPGS",size=12)
plt.plot(N_list_epgs_2d,mse_list_epgs_2d,label=r'2d case epgs', linestyle="--",color="blue")
plt.plot(N_list_epgs_5d,mse_list_epgs_5d,label=r'5d case epgs', linestyle="-",color="red")
plt.legend(loc="upper right",fontsize=14)
plt.xlabel("$\mu$",size=20)
plt.tick_params(labelsize=15)
fig_path = "C:/Users/HP/Desktop/Optimization with Objective Trans and Guassian Smoothing/Drafts/Figures/"
#plt.savefig(fig_path+"Fig2-epgs-mse.eps",bbox_inches='tight', transparent=True) #
plt.show()

#Plot
#pgs
plt.figure(figsize=(6,5))
plt.title("$f(\mu)$ - EPGS",size=12)
plt.plot(N_list_pgs_2d,fitness_list_pgs_2d,label=r'2d case pgs', linestyle="--",color="blue")
plt.plot(N_list_pgs_5d,fitness_list_pgs_5d,label=r'5d case pgs', linestyle="-",color="red")
plt.legend(loc="lower right",fontsize=14)
plt.xlabel("$\mu$",size=20)
plt.tick_params(labelsize=15)
fig_path = "C:/Users/HP/Desktop/Optimization with Objective Trans and Guassian Smoothing/Drafts/Figures/"
#plt.savefig(fig_path+"Fig2-pgs-fit.eps",bbox_inches='tight', transparent=True) #
plt.show()

plt.figure(figsize=(6,5))
plt.rcParams['text.usetex'] = True
plt.title(r"$mse(\mathbf{x}^*,\mu),$ - EPGS",size=12)
plt.plot(N_list_pgs_2d,mse_list_pgs_2d,label=r'2d case pgs', linestyle="--",color="blue")
plt.plot(N_list_pgs_5d,mse_list_pgs_5d,label=r'5d case pgs', linestyle="-",color="red")
plt.legend(loc="upper right",fontsize=14)
plt.xlabel("$\mu$",size=20)
plt.tick_params(labelsize=15)
fig_path = "C:/Users/HP/Desktop/Optimization with Objective Trans and Guassian Smoothing/Drafts/Figures/"
#plt.savefig(fig_path+"Fig2-epgs-mse.eps",bbox_inches='tight', transparent=True) #
plt.show()


# Create a 2x2 grid of subplots ---------- Figure 4
plt.rcParams['text.usetex'] = True
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# First plot (Top-left)
axs[0, 0].set_title("Fitness $f(\mathbf{\mu})$ - EPGS", size=28)
axs[0, 0].plot(N_list_epgs_2d, fitness_list_epgs_2d, label=r'EPGS-2D case', linestyle="--", color="blue")
axs[0, 0].plot(N_list_epgs_5d, fitness_list_epgs_5d, label=r'EPGS-5D case', linestyle="-", color="red")
axs[0, 0].legend(loc="center right", fontsize=22)
axs[0, 0].set_xlabel("$N$", size=22)
axs[0, 0].tick_params(labelsize=22)

# Second plot (Top-right)
axs[1, 0].set_title("MSE$(\mathbf{\mathit{m}}_1,\mu),$ - EPGS", size=28)
axs[1, 0].plot(N_list_epgs_2d, mse_list_epgs_2d, label=r'EPGS-2D case', linestyle="--", color="blue")
axs[1, 0].plot(N_list_epgs_5d, mse_list_epgs_5d, label=r'EPGS-5D case', linestyle="-", color="red")
axs[1, 0].legend(loc="upper right", fontsize=22)
axs[1, 0].set_xlabel("$N$", size=22)
axs[1, 0].tick_params(labelsize=22)

# Third plot (Bottom-left)
axs[0, 1].set_title(r"Fitness $f(\mathbf{\mu})$ - PGS", size=28)
axs[0, 1].plot(N_list_pgs_2d, fitness_list_pgs_2d, label=r'PGS-2D case', linestyle="--", color="blue")
axs[0, 1].plot(N_list_pgs_5d, fitness_list_pgs_5d, label=r'PGS-5D case', linestyle="-", color="red")
axs[0, 1].legend(loc="center right", fontsize=22)
axs[0, 1].set_xlabel("$N$", size=22)
axs[0, 1].tick_params(labelsize=22)

# Fourth plot (Bottom-right)
axs[1, 1].set_title(r"MSE$(\mathbf{\mathit{m}}_1,\mu)$ - PGS", size=28)
axs[1, 1].plot(N_list_pgs_2d, mse_list_pgs_2d, label=r'PGS-2D case ', linestyle="--", color="blue")
axs[1, 1].plot(N_list_pgs_5d, mse_list_pgs_5d, label=r'PGS-5D case', linestyle="-", color="red")
axs[1, 1].legend(loc="upper right", fontsize=22)
axs[1, 1].set_xlabel("$N$", size=22)
axs[1, 1].tick_params(labelsize=22)

# Adjust layout
plt.tight_layout()

# Save the figure
#fig_path = "C:/Users/HP/Desktop/Optimization with Objective Trans and Guassian Smoothing/Drafts/Figures/"
plt.savefig("Fig2-PowerEffect.eps", bbox_inches='tight', transparent=True)

# Show the plot
plt.show()