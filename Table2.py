from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
import numpy as np
import pandas as pd
import os,gc
import matplotlib.pyplot as plt
from matplotlib import cm
#from mpl_toolkits.mplot3d import axes3d
#from matplotlib.axes import Axes as ax
#from scipy.optimize import minimize
from Homotopy_Methods import ZOSLGH_d, ZOSLGH_r, STD_Homotopy
from Smoothing_Methods import exp_gs_adapt, ZO_AdaMM, ZO_SGD, evaluate_gs, power_gs_baseline
from Objective_functions import Rosenbrock, plot_Rosenbrock, Ackley, plot_Ackley, objective_2log, plot_obj_2log
import cma



############## 
#Set common parameters for compared algos
dim_num = 2
generation_num = 1000
repeat_num = 100
pop_size = 100 #number of randomly sampled points used for each parameter update
true_sol = np.array([1.0, 1.0])
smooth_param_candidates = [1.0,2.0] 
learning_rates = [0.2, 0.1, 0.01, 0.001, 0.0001]
objective = Rosenbrock
initial_guesses = np.random.multivariate_normal(mean=[-3.0,2.0],cov=0.01*np.eye(2),size=repeat_num)

#exponential Gaussian smoothing
final_fit = -1000
final_solution = [-5,-5]
final_mse = 1e5
final_index = -1
final_params = {"lr":-1,"smooth_param":-1,"N":-1}

for N in [1,2,3]:
    for lr in learning_rates:#learning_rates:
        for sp in smooth_param_candidates:
            mu_times = [] #number of iterations taken to achieve the best mu
            mu_fit = []
            mse_list = []
            mu_list = []
            
            for init_guess in initial_guesses:
                mu_log = exp_gs_adapt(
                            init_mu=init_guess, 
                            dim=dim_num, 
                            fitness_fcn=objective,
                            init_lr=lr, 
                            sigma=sp,
                            power=N,
                            sga_sample_size=pop_size,
                            total_step_limit=generation_num
                )
                
                best_index, best_sol, best_fit, best_mse = evaluate_gs(mu_log, objective, 
                                                                   true_sol, verbose=True)
                mu_times.append(best_index) #number of iterations taken to achieve the best mu
                mu_fit.append(best_fit)
                mse_list.append(best_mse) 
                mu_list.append(best_sol)
                
            print(" ")
            #print("N:",N, "init_lr:", lr, "smoothing_param:",sp)
            
            
            if np.mean(mse_list)<final_mse:
                final_solution = np.mean(mu_list,axis=0)
                final_index = np.mean(mu_times)
                final_mse = np.mean(mse_list)
                final_fit = np.mean(mu_fit)
                final_params["N"] = N
                final_params["lr"] = lr
                final_params["smooth_param"] = sp

print(" ")
print("Final Solution:", final_solution.round(3))
print("Fitness:", round(final_fit,3))
print("Time to best:", final_index)
print("mse:", final_mse)
print(final_params)


#power Gaussian smoothing
final_fit = -1e20
final_solution = [-10.0,-10.0]
final_mse = 1e5
final_index = -1
final_params = {"lr":-1,"smooth_param":-1,"N":-1}

def pgs_objective(sol):
    return objective(sol)+12000

for N in [1,3,5,]:
    for lr in learning_rates:
        for sp in smooth_param_candidates:
            mu_times = [] #number of iterations taken to achieve the best mu
            mu_fit = []
            mse_list = []
            mu_list = []
            
            for init_guess in initial_guesses:
                mu_log = power_gs_baseline(
                            init_mu=init_guess, 
                            dim=dim_num, 
                            fitness_fcn=pgs_objective,
                            init_lr=lr, 
                            sigma=sp,
                            power=N,
                            sga_sample_size=pop_size,
                            total_step_limit=generation_num
                )
                #print("N:",N, "init_lr:", lr, "smoothing_param:",sp)
                best_index, best_sol, best_fit, best_mse = evaluate_gs(mu_log, objective, 
                                                                   true_sol, verbose=True)
                mu_times.append(best_index) #number of iterations taken to achieve the best mu
                mu_fit.append(best_fit)
                mse_list.append(best_mse) 
                mu_list.append(best_sol)
                
            #print(" ")
            if np.mean(mse_list)<final_mse:
                final_solution = np.mean(mu_list,axis=0)
                final_index = np.mean(mu_times)
                final_mse = np.mean(mse_list)
                final_fit = np.mean(mu_fit)
                final_params["N"] = N
                final_params["lr"] = lr
                final_params["smooth_param"] = sp

print(" ")
print("Final Solution:", final_solution.round(3))
print("Fitness:", round(final_fit,3))
print("Time to best:", final_index)
print("mse:", final_mse)
print(final_params)


#Standard Homotopy
final_fit = -1e20
final_solution = [-10.0,-10.0]
final_mse = 1e5
final_index = -1
final_params = {"learning_rate":-1, "init_smooth_param":-1, "sigma_decay_factor":-1}
for lr in learning_rates:
    for sp in smooth_param_candidates:
        for decay_fac in [0.2, 0.5, 0.8]:
            mu_times = [] #number of iterations taken to achieve the best mu
            mu_fit = []
            mse_list = []
            mu_list = []
                
            for init_guess in initial_guesses:
                sol_log = STD_Homotopy(
                init_mu = init_guess, 
                dim=dim_num, 
                fitness_fcn=objective,
                init_lr=lr, 
                init_sigma=sp,
                sga_sample_size=pop_size,
                total_step_limit=generation_num, 
                sigma_decay_factor=decay_fac,
                sga_step_limit=500, 
                sga_tolerance=100, 
                sigma_tolerance=10
                )
                #print("learning rate:",lr, "decay fac:",decay_fac,"init_smooth_param:",sp)
                best_index, best_sol, best_fit, best_mse = evaluate_gs(sol_log, objective, 
                                                                       true_sol, verbose=True)
                mu_times.append(best_index) #number of iterations taken to achieve the best mu
                mu_fit.append(best_fit)
                mse_list.append(best_mse) 
                mu_list.append(best_sol)
                    
            if np.mean(mse_list)<final_mse:
                final_solution = np.mean(mu_list,axis=0)
                final_index = np.mean(mu_times)
                final_mse = np.mean(mse_list)
                final_fit = np.mean(mu_fit)
                final_params["learning_rate"] = lr
                final_params["init_smooth_param"] = sp
                final_params["sigma_decay_factor"] = decay_fac
            print(" ")

print(" ")
print("Final Solution:", final_solution.round(3))
print("Fitness:", round(final_fit,3))
print("Time to best:", final_index)
print("mse:", final_mse)
print(final_params)


#ZOSLGH_d
final_fit = -1e20
final_solution = [-10.0,-10.0]
final_mse = 1e5
final_index = -1
final_params = {"beta":-1, "eta":-1, "smooth_param":-1, "gamma":-1}
for beta in learning_rates:
    for eta in [0.1,0.01,0.001]:
        for gamma in [0.99,0.999,0.995]:
            for sp in smooth_param_candidates:
                mu_times = [] #number of iterations taken to achieve the best mu
                mu_fit = []
                mse_list = []
                mu_list = []
            
                for init_guess in initial_guesses:
                    sol_log = ZOSLGH_d(iter_num=generation_num, 
                               f=objective, 
                               init_x=init_guess,  
                               init_sigma=sp, 
                               beta=beta,#lr, 
                               eta=eta,#0.01,#eta, 
                               sample_num=pop_size, 
                               gamma=gamma, 
                               epsilon=0.001)
                    #print("beta:",lr, "eta:",eta,"smooth_param:",sp)
                    best_index, best_sol, best_fit, best_mse = evaluate_gs(sol_log, objective, 
                                                                       true_sol, verbose=True)
                    mu_times.append(best_index) #number of iterations taken to achieve the best mu
                    mu_fit.append(best_fit)
                    mse_list.append(best_mse) 
                    mu_list.append(best_sol)
                    
                if np.mean(mse_list)<final_mse:
                    final_solution = np.mean(mu_list,axis=0)
                    final_index = np.mean(mu_times)
                    final_mse = np.mean(mse_list)
                    final_fit = np.mean(mu_fit)
                    final_params["beta"] = lr
                    final_params["smooth_param"] = sp
                    final_params["eta"] = eta
                    final_params["gamma"] = gamma
                print(" ")

print(" ")
print("Final Solution:", final_solution.round(3))
print("Fitness:", round(final_fit,3))
print("Time to best:", final_index)
print("mse:", final_mse)
print(final_params)


#ZOSLGH_r
final_fit = -1e20
final_solution = [-10.0,-10.0]
final_mse = 1e5
final_index = -1
final_params = {"beta":-1, "smooth_param":-1, "gamma":-1}
for beta in learning_rates:
    for gamma in [0.99,0.999,0.995]:
        for sp in smooth_param_candidates:
            mu_times = [] #number of iterations taken to achieve the best mu
            mu_fit = []
            mse_list = []
            mu_list = []
            
            for init_guess in initial_guesses:
                sol_log = ZOSLGH_r(iter_num=generation_num, 
                    f=objective, 
                    init_x=init_guess, 
                    init_sigma=1.0, 
                    beta=beta, 
                    sample_num=pop_size, 
                    gamma=gamma, 
                   )
                #print("beta:",beta,"smooth_param:",sp,"gamma:",gamma)
                best_index, best_sol, best_fit, best_mse = evaluate_gs(sol_log, objective, 
                                                                   true_sol, verbose=True)
                mu_times.append(best_index) #number of iterations taken to achieve the best mu
                mu_fit.append(best_fit)
                mse_list.append(best_mse) 
                mu_list.append(best_sol)
                
            if np.mean(mse_list)<final_mse:
                final_solution = np.mean(mu_list,axis=0)
                final_index = np.mean(mu_times)
                final_mse = np.mean(mse_list)
                final_fit = np.mean(mu_fit)
                final_params["beta"] = lr
                final_params["smooth_param"] = sp
                final_params["gamma"] = gamma
            print(" ")

print(" ")
print("Final Solution:", final_solution.round(3))
print("Fitness:", round(final_fit,3))
print("Time to best:", final_index)
print("mse:", final_mse)
print(final_params)


#ZO_AdaMM
final_fit = -1e20
final_solution = [-10.0,-10.0]
final_mse = 1e5
final_index = -1
final_params = {"lr":-1,"smooth_param":-1,"beta1":-1,"beta2":-1}

for lr in learning_rates:
    for sp in smooth_param_candidates:
        for b1 in [0.5,0.9]:
            for b2 in [0.1,0.5]:
                mu_times = [] #number of iterations taken to achieve the best mu
                mu_fit = []
                mse_list = []
                mu_list = []
                
                for init_guess in initial_guesses:
                    sol_log = ZO_AdaMM(
                              init_x=init_guess,
                              dim=dim_num, 
                              f=objective,
                              init_lr=lr, 
                              smooth_param=sp,
                              beta1=b1,
                              beta2=b2,
                              sample_size=pop_size,
                              total_step_limit=generation_num
                             )
                    best_index, best_sol, best_fit, best_mse = evaluate_gs(sol_log, objective, 
                                                                       true_sol, verbose=True)
                    mu_times.append(best_index) #number of iterations taken to achieve the best mu
                    mu_fit.append(best_fit)
                    mse_list.append(best_mse) 
                    mu_list.append(best_sol)
                
                if np.mean(mse_list)<final_mse:
                    final_solution = np.mean(mu_list,axis=0)
                    final_index = np.mean(mu_times)
                    final_mse = np.mean(mse_list)
                    final_fit = np.mean(mu_fit)
                    final_params["lr"] = lr
                    final_params["smooth_param"] = sp
                    final_params["beta1"] = b1
                    final_params["beta2"] = b2
                    
                print(" ")
                
print(" ")
print("Final Solution:", final_solution.round(3))
print("Fitness:", round(final_fit,3))
print("Time to best:", final_index)
print("mse:", final_mse)
print(final_params)



#ZO_SGD
final_fit = -1e20
final_solution = [-10.0,-10.0]
final_mse = 1e5
final_index = -1
final_params = {"lr":-1,"smooth_param":-1}

for lr in learning_rates:
    for sp in smooth_param_candidates:
        mu_times = [] #number of iterations taken to achieve the best mu
        mu_fit = []
        mse_list = []
        mu_list = []
        
        for init_guess in initial_guesses:
            sol_log = ZO_SGD(
                         init_x=init_guess, 
                         dim=dim_num, 
                         f=objective,
                         init_lr=lr, 
                         smooth_param=sp,#smoothing_parameter,
                         sample_size=pop_size,
                         total_step_limit=generation_num
                        )
            best_index, best_sol, best_fit, best_mse = evaluate_gs(sol_log, objective, 
                                                               true_sol, verbose=True)
            mu_times.append(best_index) #number of iterations taken to achieve the best mu
            mu_fit.append(best_fit)
            mse_list.append(best_mse) 
            mu_list.append(best_sol)
                
        if np.mean(mse_list)<final_mse:
            final_solution = np.mean(mu_list,axis=0)
            final_index = np.mean(mu_times)
            final_mse = np.mean(mse_list)
            final_fit = np.mean(mu_fit)
            final_params["lr"] = lr
            final_params["smooth_param"] = sp
                    
            print(" ")
                
print(" ")
print("Final Solution:", final_solution.round(3))
print("Fitness:", round(final_fit,3))
print("Time to best:", final_index)
print("mse:", final_mse)
print(final_params)



#CMA-ES
import warnings
warnings.filterwarnings("ignore", module="cma")

dim_num = 2
generation_num = 1000
repeat_num = 100
pop_size = 100 #number of randomly sampled points used for each parameter update
true_sol = np.array([1.0,1.0])
initial_guesses = np.random.multivariate_normal(mean=[-3.0,2.0],cov=0.01*np.eye(2),size=repeat_num)

final_fit = -1000
final_solution = [-5,-5]
final_mse = 1e5
final_index = -1
final_params = {"sigma0":-1,"initial_cov":-1}

for sigma0 in [0.1,0.5,1.0]:
    for initial_cov in [1.0]:#learning_rates:
        mu_times = [] #number of iterations taken to achieve the best mu
        mu_fit = []
        mse_list = []
        mu_list = []

        opts = {
        'popsize': pop_size,    
        'CMA_stds': [initial_cov]*2, # Initial C with different scales
        'verb_disp': 0
        }
            
        for init_guess in initial_guesses:
            mu_log = []  #for storing the best sol in each iteration
            cma_es = cma.CMAEvolutionStrategy(init_guess, sigma0, opts)
            for i in range(generation_num):#range():
                sol_candidates = np.array(cma_es.ask())  #generate solution_candidates using the current model parameters (e.g., the cov matrix)
                loss_values = -Rosenbrock(sol_candidates)    #Input to Ackley should be an np.2darray (a list of solutions)
                cma_es.tell(sol_candidates, loss_values) #update model parameters
                mu_log.append(cma_es.result.xbest)       #the best sol up to the current iteration
            
            best_index, best_sol, best_fit, best_mse = evaluate_gs(mu_log, Rosenbrock, 
                                                                   true_sol, verbose=True)
            mu_times.append(best_index) #number of iterations taken to achieve the best mu
            mu_fit.append(best_fit)
            mse_list.append(best_mse) 
            mu_list.append(best_sol)
            print(" ")
            
            
        if np.mean(mse_list)<final_mse:
            final_solution = np.mean(mu_list,axis=0)
            final_index = np.mean(mu_times)
            final_mse = np.mean(mse_list)
            final_fit = np.mean(mu_fit)
            final_params["sigma0"] = sigma0
            final_params["initial_cov"] = initial_cov

print(" ")
print("Final Solution:", final_solution.round(3))
print("Fitness:", round(final_fit,3))
print("Time to best:", final_index)
print("mse:", final_mse)
print(final_params)


