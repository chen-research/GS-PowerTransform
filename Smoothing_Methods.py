import numpy as np

def exp_gs(init_mu,     #np.1darray, initial value of mu.
           dim,         #int, dimention number of the mu space.  
           fitness_fcn, #the original objective function f to be maximized.
           init_lr=0.001, #float, initial learning rate for updating mu.
           sigma=1.0,     #float, the transformed problem is max_mu E[exp(N*f(mu+sigma*xi))] where xi ~ N(0,I)
           power=2,       #int, the power is N.
           sga_sample_size=1000,  #int, the number of xi samples used to approximate the gradient of the objective.
           total_step_limit=10000, #int, total number of mu updates to be performed.
           ):
    """
    Exponential Gaussian Smooth Algorithm, for optimization. This algorithm solves
             max_{mu} E_{xi ~ N(0,I)}[exp(N*f(mu+sigma*xi))]
    Specifically, exp_gs iteratively approximate the gradient of the expectation with respect to mu,
    and use this approximated gradient to update mu (using gradient ascent).

      
    Outputs.
    ---------
    mu_log:list of mu in each update, where each mu is a np.1darray.
    """
    mu = init_mu
    #logs
    mu_log = []
    for k in range(total_step_limit):
        generation = mu+np.random.normal(loc=0, scale=sigma, size=(sga_sample_size,dim))
        sol = np.append(generation, mu.reshape((1,-1)), axis=0)
        stacked_fvalues = fitness_fcn(sol)
        fitness = stacked_fvalues[0:-1]
        mu_fit = stacked_fvalues[-1]
        alpha = init_lr  #learning rate
        #v = np.exp((fitness-mu_fit)*power)  #using a baseline
        v = np.exp(fitness*power)  #exponential_fit
        gradient = np.mean( (generation-mu)*v.reshape((-1,1)), axis=0 ) #E[(X-mu)*exp(N*f(X))]
        gradient = gradient/(np.sqrt(np.sum(gradient**2))) #normalize the gradient
        mu = mu + alpha*gradient
        mu_log.append(mu)   
        
        #Print to Screen to check training statistics
        #if k%100==0:
        #    print("k:",k)
        #    print("Calini&Wagner Loss:",-mu_fit)
        #    print("exp Nf(mu_fit)",np.exp(mu_fit*power))
        #    print("mu_norm:", np.sqrt(np.sum(mu**2)))
    return mu_log


def exp_gs_baseline(init_mu,     #np.1darray, initial value of mu.
                    dim,         #int, dimention number of the mu space.  
                    fitness_fcn, #the original objective function f to be maximized.
                    init_lr=0.001, #float, initial learning rate for updating mu.
                    sigma=1.0,     #float, the transformed problem is max_mu E[exp(N*f(mu+sigma*xi))] where xi ~ N(0,I)
                    power=2,       #int, the power is N.
                    sga_sample_size=1000,  #int, the number of xi samples used to approximate the gradient of the objective.
                    total_step_limit=10000, #int, total number of mu updates to be performed.
           ):
    """
    exp_gs_baseline performs the same udpate as that of exp_gs, and it avoids computation overflow.

    Exponential Gaussian Smooth Algorithm, for optimization. This algorithm solves
             max_{mu} E_{xi ~ N(0,I)}[exp(N*f(mu+sigma*xi))]
    Specifically, exp_gs iteratively approximate the gradient of the expectation with respect to mu,
    and use this approximated gradient to update mu (using gradient ascent).

      
    Outputs.
    ---------
    mu_log:list of mu in each update, where each mu is a np.1darray.
    """
    mu = init_mu
    #logs
    mu_log = []
    for k in range(total_step_limit):
        generation = mu+np.random.normal(loc=0, scale=sigma, size=(sga_sample_size,dim))
        sol = np.append(generation, mu.reshape((1,-1)), axis=0)
        stacked_fvalues = fitness_fcn(sol)
        fitness = stacked_fvalues[0:-1]
        mu_fit = stacked_fvalues[-1]
        alpha = init_lr * 1000/(1000+k) #learning rate
        v = np.exp((fitness-mu_fit)*power)  #using a baseline
        #v = np.exp(fitness*power)  #exponential_fit
        gradient = np.mean( (generation-mu)*v.reshape((-1,1)), axis=0 ) #E[(X-mu)*exp(N*f(X))]
        gradient = gradient/(np.sqrt(np.sum(gradient**2))) #normalize the gradient
        mu = mu + alpha*gradient
        mu_log.append(mu)   
        
        #Print to Screen to check training statistics
        #if k%100==0:
        #    print("k:",k)
        #    print("Calini&Wagner Loss:",5-mu_fit)
        #    print("exp Nf(mu_fit)",np.exp(mu_fit*power))
        #    print("mu_norm:", np.sqrt(np.sum(mu**2)))
    return mu_log


def exp_gs_adapt(init_mu,     #np.1darray, initial value of mu.
           dim,         #int, dimention number of the mu space.  
           fitness_fcn, #the original objective function f to be maximized.
           init_lr=0.001, #float, initial learning rate for updating mu.
           sigma=1.0,     #float, the transformed problem is max_mu E[exp(N*f(mu+sigma*xi))] where xi ~ N(0,I)
           power=2,       #int, the power is N.
           sga_sample_size=1000,  #int, the number of xi samples used to approximate the gradient of the objective.
           total_step_limit=10000, #int, total number of mu updates to be performed.
           ):
    """
    The only difference between exp_gs_adapt and exp_gs is, the former uses an adaptive learning rate
    of alpha = init_lr * 1000/(1000+k), while the latter uses a fixed learning rate init_lr.

    Exponential Gaussian Smooth Algorithm, for optimization. This algorithm solves
             max_{mu} E_{xi ~ N(0,I)}[exp(N*f(mu+sigma*xi))]
    Specifically, exp_gs iteratively approximate the gradient of the expectation with respect to mu,
    and use this approximated gradient to update mu (using gradient ascent).

      
    Outputs.
    ---------
    mu_log:list of mu in each update, where each mu is a np.1darray.
    """
    mu = init_mu
    #logs
    mu_log = []
    for k in range(total_step_limit):
        generation = mu+np.random.normal(loc=0, scale=sigma, size=(sga_sample_size,dim))
        sol = np.append(generation, mu.reshape((1,-1)), axis=0)
        stacked_fvalues = fitness_fcn(sol)
        fitness = stacked_fvalues[0:-1]
        mu_fit = stacked_fvalues[-1]
        alpha = init_lr * 1000/(1000+k)  #learning rate
        #v = np.exp((fitness-mu_fit)*power)  #using a baseline
        v = np.exp(fitness*power)  #exponential_fit
        gradient = np.mean( (generation-mu)*v.reshape((-1,1)), axis=0 ) #E[(X-mu)*exp(N*f(X))]
        gradient = gradient/(np.sqrt(np.sum(gradient**2))) #normalize the gradient
        mu = mu + alpha*gradient
        mu_log.append(mu)   
        
        #Print to Screen to check training statistics
        #if k%100==0:
        #    print("k:",k)
        #    print("Calini&Wagner Loss:",-mu_fit)
        #    print("exp Nf(mu_fit)",np.exp(mu_fit*power))
        #    print("mu_norm:", np.sqrt(np.sum(mu**2)))
    return mu_log


def ZO_AdaMM(init_x,                #np.1darray, the initial value of the solution candidate.
             dim,                   #int, dimention of x. (dimension of the search space). 
             f,                     #the objective function.
             init_lr=0.001,         #initial learning rate.
             smooth_param=0.1,      #the smoothing parameter.
             beta1=0.9,             #parameter for computing the x-update term.
             beta2=0.3,             #parameter for computing the x-update term.
             sample_size=10,        #number of xi samples used when approximating the gradient.
             total_step_limit=10000,  #number of x-updates to be performed.
             ):
    """
    It performs the max-version of the ZO-AdaMM algorithm to solve 
                max_{x} E_{xi~N(0,I)} [f(X+smooth_param*xi)],
    where N(0,I) is a multivariate standard normal distribution.

    Reference.
    Algorithm 1
    ZO-AdaMM: Zeroth-Order Adaptive Momentum Method for Black-Box Optimization
    Xiangyi Chen, Sijia Liu, Kaidi Xu, Xingguo Li, Xue Lin, Mingyi Hong, David Cox

    Outputs.
    --------
    x_log:list of x candidates found in each x-update.

    """
    xt = init_x
    
    mt_lag = np.zeros(dim,dtype=float)
    vt_lag = np.zeros(dim,dtype=float)
    vt_hat_lag = np.zeros(dim,dtype=float)
    
    x_log = []
    for k in range(total_step_limit):
        u = uniform_sphere(center=np.zeros(dim,dtype=float), 
                           radius=1, dim_num=dim, points_num=sample_size)
        sol = np.append(xt+smooth_param*u,xt.reshape([1,-1]), axis=0)
        f_values = f(sol)
        fcn_values = np.reshape(f_values[0:-1]-f_values[-1], [-1,1])  #f(x+smooth_param*u)-f(x)
        gt = (dim/smooth_param)*np.mean(fcn_values*u,axis=0) #gradient estimate
        mt = beta1*mt_lag + (1-beta1)*gt
        vt = beta2*vt_lag + (1-beta2)*(gt**2)
        vt_hat = np.max( np.append(vt_hat_lag.reshape([-1,1]),
                                   vt.reshape([-1,1]),
                                   axis=1), 
                         axis=1
                       )
        xt = xt + init_lr*(1000/(1000+k)) * mt/np.sqrt(vt_hat)

        mt_lag = mt.copy()
        vt_lag = vt.copy()
        vt_hat_lag = vt_hat.copy()
        x_log.append(xt)

    return x_log


def uniform_sphere(center, radius, dim_num, points_num):
    """
    Uniformly draw points from within a sphere.
    
    Inputs
    -------
    center:np.1darray, center of the sphere.
    radius:float, radius of the sphere.
    dim_num: dimension number of the space in which the sphere is in.
    points_num:int, number of points to be drawn.
    
    Outputs
    -------
    points:np.2darray, of dimension [size, dim_num].
    """
    v = np.random.normal(loc=0.0,scale=1.0,size=(points_num,dim_num))
    v_norm = np.sqrt(np.sum(v**2,axis=1))
    non_zero_indices = np.where(v_norm>0)[0]
    non_zero_v_num = len(non_zero_indices)
    v = v[non_zero_indices]  #remove v that is zero
    v_norm = v_norm[non_zero_indices]  
    v = v/np.reshape(v_norm, (-1,1)) #uniformly distributed on a unit ball
    radii = np.random.uniform(low=0.0,high=radius,size=(non_zero_v_num,1))
    v = v*radii #uniformly distributed in a zero-center sphere with radius equal to radius
    points = np.reshape(center,[1,-1]) + v
    
    return points



def ZO_SGD(init_x,                  #np.1darray, the initial value of the solution candidate.
             dim,                   #int, dimention of x. (dimension of the search space). 
             f,                     #the objective function.
             init_lr=0.001,         #initial learning rate for updating x.
             smooth_param=0.1,      #the smoothing parameter.
             sample_size=10,        #number of xi samples used when approximating the gradient.
             total_step_limit=10000, #number of x-updates to be performed.
             ):
    """
    It performs the max-version of the zeroth-order randomized stochastic gradient (RSG) method to solve 
                max_{x} E_{xi~N(0,I)} [f(X+smooth_param*xi)],
    where N(0,I) is a multivariate standard normal distribution.

    Reference.
    Section 2.1
    STOCHASTIC FIRST- AND ZEROTH-ORDER METHODS FOR NONCONVEX STOCHASTIC PROGRAMMING
    SAEED GHADIMI AND GUANGHUI LAN

    Outputs.
    --------
    x_log:list of x candidates found in each x-update.

    """

    xt = init_x
    x_log = []
    for k in range(total_step_limit):
        u = uniform_sphere(center=np.zeros(dim,dtype=float), 
                           radius=1, dim_num=dim, points_num=sample_size)
        sol = np.append(xt+smooth_param*u,xt.reshape([1,-1]), axis=0)
        f_values = f(sol)
        fcn_values = np.reshape(f_values[0:-1]-f_values[-1], [-1,1])  #f(x+smooth_param*u)-f(x)
        gt = (dim/smooth_param)*np.mean(fcn_values*u,axis=0) #gradient estimate
        xt = xt + init_lr*(1000/(1000+k)) * gt
        x_log.append(xt)

    return x_log




def power_gs_baseline(init_mu, dim, fitness_fcn, 
             init_lr=0.001, sigma=1.0, power=2,
             sga_sample_size=1000,
             total_step_limit=10000, 
             verbose=False):
    """
    Powered Gaussian Smooth with a Baseline Algorithm, for optimization.
    
    Inputs.
    ---------
    init_generation:np.2darray, each row is a candidate solution.
    fitness_fcn, the fitness function.
    elite_ratio:float, the portion of best candidates in each generation used to produce a Gaussian distribution.
    update_weight:float, mu = (1-update_weight)*mu + update_weight*updated_mu
    generation_num:number of generations to be generated.
    verbose:binary, whether to show the training process.
    
    Outputs.
    ---------
    best_solution:np.1darray, the solution with the largest fitness value among all the generations.
    best_fitness:float, fitness of the best solution.
    """
    mu = init_mu
    #logs
    mu_log = []
    for k in range(total_step_limit):
        generation = np.random.multivariate_normal(mean=mu, cov=sigma*np.identity(dim),
                                                   size=sga_sample_size)
        fitness = fitness_fcn(generation)
        mu_fit = fitness_fcn(mu.reshape([1,-1]))[0]
        ############ - Update mu
        alpha = init_lr * 5000/(5000+k) #learning rate
        v = (fitness/mu_fit)**power
        #v = (fitness)**power
        gradient = np.mean( (generation-mu)*v.reshape((-1,1)), axis=0 ) #E[(X-mu)*f(X)]
        gradient = gradient/(np.sqrt(np.sum(gradient**2))) #normalize the gradient
        mu = mu + alpha*gradient
        """ #print to screen
        if (k%1000)==0:
            print("total_step_counter:",k)
            print('mu',mu[0:3].round(3),"gradient",gradient[0:3].round(3))
            #print('mu_fitness:',round(mu_fit,3))
            print("v=mean( (fit(x)/fit(mu))^power )=", round(np.mean(v), 5))
        """
        mu_log.append(mu)   
    return mu_log

def power_gs(init_mu, dim, fitness_fcn, 
             init_lr=0.001, sigma=1.0, power=2,
             sga_sample_size=1000,
             total_step_limit=10000, 
             verbose=False):
    """
    Powered Gaussian Smooth with a Baseline Algorithm, for optimization.
    
    Inputs.
    ---------
    init_generation:np.2darray, each row is a candidate solution.
    fitness_fcn, the fitness function.
    elite_ratio:float, the portion of best candidates in each generation used to produce a Gaussian distribution.
    update_weight:float, mu = (1-update_weight)*mu + update_weight*updated_mu
    generation_num:number of generations to be generated.
    verbose:binary, whether to show the training process.
    
    Outputs.
    ---------
    best_solution:np.1darray, the solution with the largest fitness value among all the generations.
    best_fitness:float, fitness of the best solution.
    """
    mu = init_mu
    #logs
    mu_log = []
    for k in range(total_step_limit):
        generation = np.random.multivariate_normal(mean=mu, cov=sigma*np.identity(dim),
                                                   size=sga_sample_size)
        fitness = fitness_fcn(generation)
        mu_fit = fitness_fcn(mu.reshape([1,-1]))[0]
        ############ - Update mu
        alpha = init_lr * 5000/(5000+k) #learning rate
        #v = (fitness/mu_fit)**power
        v = (fitness)**power
        gradient = np.mean( (generation-mu)*v.reshape((-1,1)), axis=0 ) #E[(X-mu)*f(X)]
        gradient = gradient/(np.sqrt(np.sum(gradient**2))) #normalize the gradient
        mu = mu + alpha*gradient
        
        """#print to screen
        if (k%1000)==0:
            print("total_step_counter:",k)
            print('mu',mu[0:3].round(3),"gradient",gradient[0:3].round(3))
            #print('mu_fitness:',round(mu_fit,3))
            print("v=mean( (fit(x)/fit(mu))^power )=", round(np.mean(v), 5))
        """
        mu_log.append(mu)   
    return mu_log





def evaluate_gs(solution_log, f, true_solution, verbose=False):
    """
    Output the solution in solution_log that is closest to true_solution.
    
    Inputs.
    --------
    solution_log:list of solutions, each row in solution_log is a solution.
    f: the fitness function.
    true_solution:1darray, the true solution to max f(x).

    Outputs.
    --------
    best_sol:1darray, the best solution in solution_log.
    best_fit:float, the fitness value of the best solution.
    
    """
    mse_arr = np.mean((np.array(solution_log)-true_solution)**2,axis=1) # arrays of mse
    best_fit_index = np.argmin(mse_arr)
    best_sol = solution_log[best_fit_index]
    best_fit = f(best_sol.reshape([1,-1]))[0]
    if verbose:
        print("best fitness:",round(best_fit,3))
        print("Steps taken to reach the best fit:",best_fit_index)
        print("best solution:",best_sol.round(3))
        print("mse of best solution",mse_arr[best_fit_index])
    
    #fitness_values = f(np.array(solution_log))
    #plt.figure()
    #plt.plot(range(1,1+len(solution_log)),fitness_values)
    #plt.title("Fitness value of the solution found in each step.")
    #plt.xlabel("Step Number")
               
    return [best_fit_index, best_sol, best_fit, mse_arr[best_fit_index]]