import numpy as np

def ZOSLGH_d(iter_num,   #int, the total interation number (both x and sigma are updated once in each iteration) 
         f,          #the original objective function to maximize
         init_x,     #float, initial solution guess
         init_sigma, #float, initial value of the scaling parameter (cov=sigma*I)
         beta,       #float, learning rate for x
         eta,        #float, learning rate for sigma
         sample_num=1000,  #int, number of samples used to compute the gradient estimate
         gamma=0.95, #float, a number used to control size of sigma
         epsilon=0.01 #float, a small number used to control size of sigma
    ):
    """
    The max-version of the zeroth-order single-loop Gaussian homotopy (deterministic version), 
    for solving max_{x} f(x). The scaling parameter is udpated using a derivative estimate: gt.
    
    Reference
    -----------
    H. Iwakiri, Y. Wang, S. Ito, and A. Takeda, 
    Single Loop Gaussian Homotopy Method for Non-convex Optimization (NeurIPS, 2022)
    Algorithm 3 
    
    Outputs.
    -----------
    x_logs:list of solution candidate x found in each x-update, where x is a np.1darray.

    Example Usage
    -----------
    solution = SLGH(iter_num=1000, f=objective, init_x=[0.1]*2, init_sigma=1.0, lr_x=0.01, 
                    lr_sigma=0.01, sample_num=1000, gamma=0.95, epsilon=0.01)
    """
    x = init_x
    sigma = init_sigma
    dim_num = len(init_x)
    x_logs = np.zeros((iter_num,dim_num), dtype=float)  #x values found in each iteration
    
    for k in range(iter_num):
        #Prepare terms for computing both gx and gt
        noise = np.random.normal(loc=0,scale=1,size=(int(sample_num*2),dim_num))
        sol = np.append(x+sigma*noise, x.reshape([1,-1]), axis=0)
        stacked_fvalues = f(sol)
        #gx = estimate of gradient with respect to x
        u = noise[0:sample_num]
        fcn_values = stacked_fvalues[0:sample_num] - stacked_fvalues[-1] #f(x+sigma*u)-f(x)
        fcn_values = fcn_values.reshape((-1,1))
        gx = np.mean(fcn_values*u,axis=0)/sigma
        x = x+beta*gx
                
        #gt = estimate of gradient with respect to sigma
        v = noise[sample_num:]
        fcn_values = stacked_fvalues[sample_num:int(sample_num*2)] - stacked_fvalues[-1] #f(x+sigma*v)-f(x))
        gt = (np.sum(v**2,axis=1)-dim_num)*fcn_values/(sigma**2) #Eq (9) in Iwakiri et al. 2022
        gt = np.mean(gt)
        sigma = min(sigma+eta*gt, gamma*sigma)
        sigma = max(sigma, epsilon)
        
        #log res
        x_logs[k,:] = x.copy()
        
    return x_logs



def ZOSLGH_r(iter_num,   #int, the total interation number (both x and sigma are updated once in each iteration) 
         f,          #the original objective function to maximize
         init_x,     #float, initial solution guess
         init_sigma, #float, initial value of the scaling parameter (cov=sigma*I)
         beta,       #float, learning rate for x, which is beta in the paper
         sample_num=1000,  #int, number of samples used to compute the gradient estimate
         gamma=0.95, #float, a number used to control size of sigma
    ):
    """
    The max-version of the zeroth-order single-loop Gaussian homotopy (deterministic version), 
    for solving max_{x} f(x). The scaling parameter is udpated using a fixed rate: sigma=sigma*gamma.
    
    Reference
    -----------
    Algorithm 3
    Single Loop Gaussian Homotopy Method for Non-convex Optimization (NeurIPS, 2022)
    H. Iwakiri, Y. Wang, S. Ito, and A. Takeda, 
    
    Outputs.
    -----------
    x_logs:list of solution candidate x found in each x-update, where x is a np.1darray.

    Example Usage
    -----------
    solution = SLGH(iter_num=1000, f=objective, init_x=[0.1]*2, init_sigma=1.0, lr_x=0.01, 
                    lr_sigma=0.01, sample_num=1000, gamma=0.95, epsilon=0.01)
    """
    x = init_x
    sigma = init_sigma
    dim_num = len(init_x)
    x_logs = np.zeros((iter_num,dim_num), dtype=float)  #x values found in each iteration
    
    for k in range(iter_num):
        #Prepare terms for computing both gx and gt
        noise = np.random.normal(loc=0,scale=1,size=(int(sample_num*2),dim_num))
        sol = np.append(x+sigma*noise, x.reshape([1,-1]), axis=0)
        stacked_fvalues = f(sol)
        #gx = estimate of gradient with respect to x
        u = noise[0:sample_num]
        fcn_values = stacked_fvalues[0:sample_num] - stacked_fvalues[-1] #f(x+sigma*u)-f(x)
        fcn_values = fcn_values.reshape((-1,1))
        gx = np.mean(fcn_values*u,axis=0)/sigma
        x = x+beta*gx
                
        #gt = estimate of gradient with respect to sigma
        v = noise[sample_num:]
        fcn_values = stacked_fvalues[sample_num:int(sample_num*2)] - stacked_fvalues[-1] #f(x+sigma*v)-f(x))
        gt = (np.sum(v**2,axis=1)-dim_num)*fcn_values/(sigma**2) #Eq (9) in Iwakiri et al. 2022
        gt = np.mean(gt)
        sigma = gamma*sigma
        
        #log res
        x_logs[k,:] = x.copy()
        
    return x_logs



#Standard Homotopy
def STD_Homotopy(init_mu, dim, fitness_fcn, 
                   init_lr=0.001, init_sigma=1.0,
                   sga_sample_size=1000,
                   total_step_limit=10000, 
                   sigma_decay_factor=0.5,
                   sigma_update_num=3,
                   sga_step_limit=1000, 
                   sga_tolerance=10, 
                   sigma_tolerance=3
                   ):
    """
    This function performs a standard homotopy algorithm for optimization.
    It performs stochastic gradient ascent to update x in E[f(x+delta*v)], where x is deterministic, 
    delta is fixed, v is a standard multivariate Gaussian noise. 
    The gradient estimate is a sample estimate of 
                   E[f(x+delta*v)*(delta*v)] = partial E[f(x+delta*v)]/partial x,
    where E is over the expectation of v and v is uniformly sampled from the 
    standard multivariate Gaussian distribution.
    
    Inputs.
    ---------
    init_mu:np.1darray, initial guess of the soluiton x.
    fitness_fcn, the fitness function. It takes in a 2darray, each row of which is an x-instance. 
                 It outputs a 1darray, and its ith entry equals to the fitness of the ith x-instance.
    dim:int, the dimension number of x.
    sga_sample_size: number of points (i.e., v) sampled for computing each gradient estimate.
    total_step_limit: the maximum number of solution (mu) updates to be performed.
    sigma_decay_factor:float, the factor used to decay the scaling parameter.
    sigma_update_num:int, maximum number of times the scaling parameter (sigma) gets decayed.
    sga_step_limit:int, maximum number of iterations (solution updates) to be performed for each fixed sigma.
    sga_tolerance:int, if the fitness do not improve for sga_tolerance solution updates, then decay sigma.
    
    Outputs.
    ---------
    mu_log:np.2darray, the history of updated solutions as the algorithm is executed.
    """
    mu = init_mu
    sigma = init_sigma
    
    #logs
    total_step_counter = 0
    sigma_update_counter = 0
    mu_log = []
    sigma_fitness_log = []
    sigma_update_keep_going = True
    
    while (total_step_counter<total_step_limit)&(sigma_update_keep_going): #outer loop for sigma update
        sga_step = 0
        sga_keep_going = True
        mu_fitness_log = []
        while ((sga_step<sga_step_limit)&(sga_keep_going)&(total_step_counter<total_step_limit)): #inner loop for mu update
            #My way to compute the gradient
            u = np.random.normal(loc=0, scale=1, size=(sga_sample_size,dim))
            generation = mu+sigma*u
            sol = np.append(generation, mu.reshape((1,-1)), axis=0)
            stacked_fvalues = fitness_fcn(sol)
            fitness = stacked_fvalues[0:-1].reshape((-1,1))
            mu_fit = stacked_fvalues[-1]
            
            ############ - Update mu
            alpha = init_lr * 500/(500+total_step_counter) #learning rate
            gradient = np.mean( (generation-mu)*fitness, axis=0 ) #E[(X-mu)*f(X)]
            gradient = gradient/(np.sqrt(np.sum(gradient**2))) #normalize the gradient
            mu = mu + alpha*gradient
            
            # mu performance
            mu_fitness_log.append(mu_fit)
            mu_log.append(mu)  #log mu
            
            #print to screen
            #if (total_step_counter%200)==0:
            #    print("total_step_counter:",total_step_counter)
            #    #print("mu",mu.round(3)[0:2],
            #           "mu_norm", round(np.sqrt(np.sum(mu**2)),3))
            
            #    print('calini&wagner loss', "mu:", -mu_fit)
            #    print("sigma:",round(sigma,5))
            #    print(" ")
            
            #If no improvement in mu_fit for tolerance steps, end sga
            if sga_step>sga_tolerance:
                if np.max(mu_fitness_log[-(sga_tolerance-1):])<=mu_fitness_log[-sga_tolerance]:
                    sga_keep_going = False
                    #print("sga_ends_at_step:",sga_step)
                    
            sga_step += 1
            total_step_counter+=1
            
        #Inner while loop ends and Update sigma
        sigma_fitness_log.append(mu_fit)
        sigma = sigma*sigma_decay_factor
        sigma_update_counter+=1
        #Determine whether to end the sigma update
        if sigma_update_counter>=sigma_tolerance:
            if np.max(sigma_fitness_log[-(sigma_tolerance-1):])<=sigma_fitness_log[-sigma_tolerance]:
                sigma_update_keep_going = False
    #print("total_step_counts",total_step_counter)
    
    return mu_log
