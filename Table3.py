from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
import numpy as np
import pandas as pd
import os,gc
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
from matplotlib.axes import Axes as ax
from Smoothing_Methods import exp_gs, ZO_AdaMM, ZO_SGD
from Homotopy_Methods import ZOSLGH_d, ZOSLGH_r, STD_Homotopy
import tensorflow as tf
from train_mnist_nn_new import MNIST,train_distillation

def calini_wagner_loss(image,x,t,c_coef,classifier,kappa=-10): 
    """
    This function computes the L2 loss function defined by Calini and Wagner (2017) Section IV A
    used for targeted adversarial attack. The loss function is
    || (1/2)*(tanh(w)+1)-image ||**2 + c_coef*f( (1/2)*(tanh(w)+1) ), where
    f(Z) = max_{i!=t} Z_i - Z_t. 
    Note that the smaller f(Z) is, the larger the probability of t is.
    
    Inputs.
    --------
    x:np.ndarray, A list of perturbations, each perturbation is to be added to an image. 
                  each of which is supposed to be close to 0s.
                  dimension: [fake_image_num, flattened image dimensions]
    image:np.ndarray, of dimension (28,28)
    t:int, the true label of the image.
    c_coef:float, a hyper-param in the loss function.
    classifier. the image classifier
    
    Output.
    --------
    cw_loss:1darray of floats, the Calini&Wagner loss for each sample.
    """
    distorted_image = x.reshape((x.shape[0],28,28,1))+image.reshape((28,28,1))
    distortion_norm = np.sqrt(np.sum(x**2, axis=1)) #x is of dimension (x.shape[0],28*28)
    
    pred = classifier.predict(distorted_image,verbose=False)  
    # Create a mask to exclude the target label logits
    mask = np.ones_like(pred, dtype=bool)
    mask[:, t] = False
    label_logit = pred[:,t].copy() #target label logit
    pred = pred[mask].reshape(pred.shape[0], pred.shape[1]-1) # Remove the target label logits
    logit_diff = np.max(pred,axis=1)-label_logit #(largest logit - target label logit) 
    logit_diff[logit_diff<kappa] = kappa
    cw_loss = logit_diff+c_coef*distortion_norm  #calini_wagner_loss
    return cw_loss

#A robust cnn (distilled cnn)
d = MNIST()
mnist_nn = train_distillation(data=d, 
                              file_name="models/mnist-distilled-100", 
                              params=(64,32),
                              num_epochs=15, 
                              train_temp=100)
x_test = d.test_data
y_test = d.test_labels
test_loss, test_accuracy = mnist_nn.evaluate(d.test_data, d.test_labels)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

#Settings
c_coef = 1.0   #coefficient used when computing the loss
r = np.random.RandomState(0)
selected_image_indices = r.choice(range(len(x_test)),replace=False,size=100)
pop_size = 10  #number of sampleds used to approximate the gradient
state_dim = 28*28  #dimension of the search space (space of flatten images)
generation_num = 1500  #number of total solution updates

def evaluation(solution_log, image, classifier, t, kappa=-10):
    solution_log = np.array(solution_log)
    sol_norms = np.sqrt(np.sum(solution_log**2,axis=1))
    sol_norms_time = np.append(sol_norms.reshape((-1,1)),
                           np.arange(0,len(solution_log)).reshape((-1,1)),
                           axis=1)
    distorted_images = solution_log.reshape((-1,28,28,1))+image.reshape([28,28,1])
    pred = classifier.predict(distorted_images)
    logit_diff = np.max(pred[:,list(range(0,t))+list(range(t+1,10))],axis=1)-pred[:,t]
    sucess_indicators = (logit_diff<=kappa)
    if np.sum(sucess_indicators)==0:
        print("No successful solutions!")
        return [np.nan]*3
    sol_norms_time = sol_norms_time[sucess_indicators]
    best_sol_index_among_success_sols = np.argmin(sol_norms_time[:,0]) #index of best sol among the sucessful solutions
    best_sol_index = int(sol_norms_time[best_sol_index_among_success_sols,1])  #number of iterations passed til the best sol
    #statistics of the best solution
    best_sol = solution_log[best_sol_index]
    logit_diff_of_best = logit_diff[best_sol_index]
    best_cost = calini_wagner_loss(image=image,x=best_sol.reshape([1,-1]),t=t,
                                   c_coef=c_coef,classifier=classifier,kappa=kappa)
    r2 = r2_score(image.flatten(),distorted_images[best_sol_index].flatten())
    print("best sol norm",round(np.sqrt(np.sum(best_sol**2)),4))
    print("Calini_Wagner Loss", best_cost)
    print("logit_diff:",logit_diff_of_best)
    print("r2 score:",r2)
    print("target label",t,"predicted label",np.argmax(pred[best_sol_index]))
    print("time to the best",best_sol_index)
    print(" ")
    return [best_sol_index, r2, logit_diff_of_best]

#ZO_SGD
logit_diff_log =[]
r2_list = []
times = [] #number of iterations taken to achieve the best mu for each image
image_count = 1
for i in selected_image_indices:
    image = x_test[i]
    label = np.argmax(y_test[i])
    pred = mnist_nn.predict(image.reshape((1,28,28,1)))[0]
    t = np.argmin(pred)
    print("image_count:", image_count, "image_num:",i, 't:',t,'true_label:',label)
    image_count += 1
    #t = r.choice(list(range(0,label))+list(range(label+1,10)))
    
    def calini_wagner_fit(x):
        loss = calini_wagner_loss(image=image,x=x,t=t,c_coef=c_coef,classifier=mnist_nn,kappa=-0.001)
        return -loss
    solution_log = ZO_SGD(init_x=np.array([0.0]*state_dim), 
                            dim=state_dim, 
                            f=calini_wagner_fit,
                            init_lr=0.0001, 
                            smooth_param=0.1, 
                            sample_size=pop_size, 
                            total_step_limit=generation_num 
                   )
    
    [best_sol_time, r2, logit_diff] = evaluation(solution_log,image,mnist_nn,t,kappa=-0.001)

    #log results
    logit_diff_log.append(logit_diff)
    r2_list.append(r2)
    times.append(best_sol_time) 

#Summary Results
zosgd_res = pd.DataFrame({
                         "r2":r2_list,"time to achieve best":times, 
                         "success":(np.array(logit_diff_log)<-0.001).astype(int),
                         "time":times
                   })

#zosgd_res.to_csv("zosgd_mnist_"+str(generation_num)+".csv",index=False)
zosgd_res.describe()

#exponential Gaussian smooth - hardest attack
N = 0.02
sol_logit_diff_log = []
sol_r2_list = []
sol_times = [] #number of iterations taken to achieve the best solution for each image
mu_logit_diff_log =[]
mu_r2_list = []
mu_times = [] #number of iterations taken to achieve the best mu for each image
image_count = 1

for i in selected_image_indices:#[[13,46,54,55,57,76]]:
    image = x_test[i]
    label = np.argmax(y_test[i])
    pred = mnist_nn.predict(image.reshape((1,28,28,1)))[0]
    t = np.argmin(pred)
    print("image count:", image_count, "image no:",i,'t:',t,"true_label:",label)
    image_count += 1
    
    #t = r.choice(list(range(0,label))+list(range(label+1,10)))
    
    def calini_wagner_fit(x):
        loss = calini_wagner_loss(image=image,x=x,t=t,c_coef=c_coef,classifier=mnist_nn,kappa=-0.001)
        return -loss
    
    #init_x = r.multivariate_normal(mean=[0.0]*state_dim,cov=np.identity(state_dim)*0.0000001,size=1)
    mu_log = mu_log = exp_gs(
                     init_mu=np.array([0]*state_dim),     #np.1darray, initial value of mu.
                     dim=state_dim,         #int, dimention number of the mu space.  
                     fitness_fcn=calini_wagner_fit, #the original objective function f to be maximized.
                     init_lr=0.1,   #float, initial learning rate for updating mu.
                     sigma=0.1,     #float, the transformed problem is max_mu E[exp(N*f(mu+sigma*xi))] where xi ~ N(0,I)
                     power=N,       #int, the power is N.
                     sga_sample_size=pop_size,  #int, the number of xi samples used to approximate the gradient of the objective.
                     total_step_limit=generation_num, #int, total number of mu updates to be performed.
                    )
        
    print("image:",i)
    mu_time, r2, logit_diff_of_best = evaluation(mu_log,image,mnist_nn,t,kappa=-0.001)
    #log results
    mu_logit_diff_log.append(logit_diff_of_best)
    mu_r2_list.append(r2)
    mu_times.append(mu_time) 

#Summary Results
expgs_res = pd.DataFrame({
                    "mu_r2":mu_r2_list,"mu_time":mu_times, 
                    "mu_success":(np.array(mu_logit_diff_log)<-0.001).astype(int),
                   })
#expgs_res.to_csv("expgs_mnist_"+str(generation_num)+".csv",index=False)
expgs_res.describe()


#zoslgh_d - hardest attack
logit_diff_log =[]
r2_list = []
times = [] #number of iterations taken to achieve the best mu for each image
image_count = 1
for i in selected_image_indices:
    image = x_test[i]
    label = np.argmax(y_test[i])
    pred = mnist_nn.predict(image.reshape((1,28,28,1)))[0]
    t = np.argmin(pred)
    print("image_count:", image_count, "image_num:",i, 't:',t,'true_label:',label)
    image_count += 1
    #t = r.choice(list(range(0,label))+list(range(label+1,10)))
    
    def calini_wagner_fit(x):
        loss = calini_wagner_loss(image=image,x=x,t=t,c_coef=c_coef,classifier=mnist_nn,kappa=-0.001)
        return -loss
    #init_x = r.multivariate_normal(mean=[0.0]*state_dim,cov=np.identity(state_dim)*0.0000001,size=1)
    solution_log = ZOSLGH_d(iter_num=generation_num, f=calini_wagner_fit, 
                    init_x=np.array([0.0]*state_dim), 
                    init_sigma=0.1, beta=0.0001, 
                    eta=0.001, sample_num=pop_size, gamma=0.995, epsilon=0.001)
    [best_sol_time, r2, logit_diff] = evaluation(solution_log,image,mnist_nn,t,kappa=-0.001)

    #log results
    logit_diff_log.append(logit_diff)
    r2_list.append(r2)
    times.append(best_sol_time) 

#Summary Results
slgh_res = pd.DataFrame({
                         "r2":r2_list,"time to achieve best":times, 
                         "success":(np.array(logit_diff_log)<-0.001).astype(int),
                         "time":times
                   })

#slgh_res.to_csv("slgh_d_mnist_"+str(generation_num)+".csv",index=False)
slgh_res.describe()

#zoslgh_r - hardest attack
logit_diff_log =[]
r2_list = []
times = [] #number of iterations taken to achieve the best mu for each image
image_count = 1
for i in selected_image_indices:
    image = x_test[i]
    label = np.argmax(y_test[i])
    pred = mnist_nn.predict(image.reshape((1,28,28,1)))[0]
    t = np.argmin(pred)
    print("image_count:", image_count, "image_num:",i, 't:',t,'true_label:',label)
    image_count += 1
    #t = r.choice(list(range(0,label))+list(range(label+1,10)))
    
    def calini_wagner_fit(x):
        loss = calini_wagner_loss(image=image,x=x,t=t,c_coef=c_coef,classifier=mnist_nn,kappa=-0.001)
        return -loss
    #init_x = r.multivariate_normal(mean=[0.0]*state_dim,cov=np.identity(state_dim)*0.0000001,size=1)
    solution_log = ZOSLGH_r(iter_num=generation_num, f=calini_wagner_fit, 
                    init_x=np.array([0.0]*state_dim), 
                    init_sigma=0.1, beta=0.0001, 
                    sample_num=pop_size, gamma=0.995)
    [best_sol_time, r2, logit_diff] = evaluation(solution_log,image,mnist_nn,t,kappa=-0.001)

    #log results
    logit_diff_log.append(logit_diff)
    r2_list.append(r2)
    times.append(best_sol_time) 

#Summary Results
slgh_res = pd.DataFrame({
                         "r2":r2_list,"time to achieve best":times, 
                         "success":(np.array(logit_diff_log)<-0.001).astype(int),
                         "time":times
                   })
#slgh_res.to_csv("slgh_r_mnist_"+str(generation_num)+".csv",index=False)
slgh_res.describe()

#ZO-AdaMM
logit_diff_log =[]
r2_list = []
times = [] #number of iterations taken to achieve the best mu for each image
image_count = 1
for i in selected_image_indices:
    image = x_test[i]
    label = np.argmax(y_test[i])
    pred = mnist_nn.predict(image.reshape((1,28,28,1)))[0]
    t = np.argmin(pred)
    print("image_count:", image_count, "image_num:",i, 't:',t,'true_label:',label)
    image_count += 1
    #t = r.choice(list(range(0,label))+list(range(label+1,10)))
    
    def calini_wagner_fit(x):
        loss = calini_wagner_loss(image=image,x=x,t=t,c_coef=c_coef,classifier=mnist_nn,kappa=-0.001)
        return -loss
    solution_log = ZO_AdaMM(init_x=np.array([0.0]*state_dim), 
                            dim=state_dim, 
                            f=calini_wagner_fit,
                            init_lr=0.1, 
                            smooth_param=0.1, 
                            beta1=0.9, 
                            beta2=0.1,
                            sample_size=pop_size, 
                            total_step_limit=generation_num 
                   )
    
    [best_sol_time, r2, logit_diff] = evaluation(solution_log,image,mnist_nn,t,kappa=-0.001)

    #log results
    logit_diff_log.append(logit_diff)
    r2_list.append(r2)
    times.append(best_sol_time) 

#Summary Results
zoadamm_res = pd.DataFrame({
                         "r2":r2_list,"time to achieve best":times, 
                         "success":(np.array(logit_diff_log)<-0.001).astype(int),
                         "time":times
                   })

#zoadamm_res.to_csv("zoadamm_mnist_"+str(generation_num)+".csv",index=False)
zoadamm_res.describe()

logit_diff_log =[]
r2_list = []
times = [] #number of iterations taken to achieve the best mu for each image
image_count = 1
for i in selected_image_indices:
    image = x_test[i]
    label = np.argmax(y_test[i])
    pred = mnist_nn.predict(image.reshape((1,28,28,1)))[0]
    t = np.argmin(pred)
    print("image_count:", image_count, "image_num:",i, 't:',t,'true_label:',label)
    image_count += 1
    #t = r.choice(list(range(0,label))+list(range(label+1,10)))
    
    def calini_wagner_fit(x):
        loss = calini_wagner_loss(image=image,x=x,t=t,c_coef=c_coef,classifier=mnist_nn,kappa=-0.001)
        return -loss
    
    solution_log = STD_Homotopy(
                    init_mu=np.array([0]*state_dim), 
                    dim=state_dim, 
                    fitness_fcn=calini_wagner_fit,
                    init_lr=0.5, 
                    init_sigma=1.0,
                    sga_sample_size=pop_size,
                    total_step_limit=generation_num, 
                    sigma_decay_factor=0.5,
                    sga_step_limit=500, 
                    sga_tolerance=100, 
                    sigma_tolerance=10
                )

    [best_sol_time, r2, logit_diff] = evaluation(solution_log,image,mnist_nn,t,kappa=-0.001)

    #log results
    logit_diff_log.append(logit_diff)
    r2_list.append(r2)
    times.append(best_sol_time) 

#Summary Results
homotopy_opt_res = pd.DataFrame({
                         "r2":r2_list,"time to achieve best":times, 
                         "success":(np.array(logit_diff_log)<-0.001).astype(int),
                         "time":times
                   })

#homotopy_opt_res.to_csv("homotopy_opt_mnist_"+str(generation_num)+".csv",index=False)
homotopy_opt_res.describe()


