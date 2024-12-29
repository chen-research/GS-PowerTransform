import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
from matplotlib.axes import Axes as ax

def objective_2log(sol):
    """
    This function computes the objective to be maximized.
    It process multiple solutions.
    
    Inputs
    --------
    sol: np.ndarray, each row is a guessed solution (an x-instance).
    dim: int, number of dimensions of the solution space.
    
    Outputs
    --------
    output: np.1darray, the ith entry is the objective value of the ith row of sol.
    """
    dim = sol.shape[1]
    m1 = np.array([-0.5]*dim) 
    m2 = np.array([0.5]*dim)
    output = -np.log(np.sum((sol-m1)**2,axis=1)+0.00001)-np.log(np.sum((sol-m2)**2,axis=1)+0.01)
    return output

def plot_obj_2log(savefig=False):
    x_delta = 0.01
    y_delta = 0.01
    x_min = -1
    x_max = 1
    y_min = -1
    y_max = 1
    x = np.arange(x_min,x_max,x_delta)
    y = np.arange(y_min,y_max,y_delta)
    dim = len(x)
    #
    sol = np.zeros((dim**2,2))
    for d in range(dim):
        sol[d*dim:(d+1)*dim,0] = x[d]
        sol[d*dim:(d+1)*dim,1] = y

    #z = np.reshape(np.exp(objective(sol)),(dim,dim))
    z = np.reshape(objective_2log(sol),(dim,dim))
    ax = plt.figure(figsize=(10,10)).add_subplot(projection='3d')
    x, y = np.meshgrid(x, y)
    # Plot the 3D surface
    #ax.title.set_text(r"$f(x)=-ln((x-m_1)^2+0.00001)-ln((x-m_2)^2+0.01)$")
    ax.plot_surface(x, y, z, cmap=cm.coolwarm, edgecolor='royalblue')#, lw=0.5, alpha=0.5)
    ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max), zlim=(np.min(z), np.max(z)))
    ax.set_xlabel('$x$', fontsize=15)
    ax.set_ylabel('$y$', fontsize=15)
    ax.tick_params(labelsize=15)
    if savefig:
        plt.savefig("objective-2log.eps", bbox_inches='tight', transparent=True)
    plt.show()




def Rosenbrock(sol):
    """
    This function computes the objective to be minimized.
    It process multiple solutions.
    
    Inputs
    --------
    sol: np.ndarray, each row is a guessed solution (an x-instance).
    dim: int, number of dimensions of the solution space.
    
    Outputs
    --------
    output: np.1darray, the ith entry is the objective value of the ith row of sol.
    """
    dim = sol.shape[1]
    output = -100*(sol[:,1]-sol[:,0]**2)**2-(1-sol[:,0])**2
    return output


def plot_Rosenbrock(savefig=False):
    #Plot the 2D objective
    x_delta = 0.01
    y_delta = 0.01
    x_min = -3
    x_max = 3
    y_min = -2
    y_max = 8
    x = np.arange(x_min,x_max,x_delta)
    y = np.arange(y_min,y_max,y_delta)
    dim_x = len(x)
    dim_y = len(y)
    #
    sol = np.zeros((dim_x*dim_y,2))
    for d in range(dim_y):
        sol[d*dim_x:(d+1)*dim_x,0] = x
        sol[d*dim_x:(d+1)*dim_x,1] = y[d]
    
    z = np.reshape(Rosenbrock(sol),(dim_y,dim_x))#+12000

    """
    # Test the correspondences
    for i in range(dim_x):
        for j in range(dim_y):
            print(Rosenbrock(np.array([[x[j,i],y[j,i]]]))[0] == z[j,i])
    """

    ax = plt.figure(figsize=(10,10)).add_subplot(projection='3d')
    x, y = np.meshgrid(x, y)
    # Plot the 3D surface
    #ax.title.set_text(r"$f(x)=-ln((x-m_1)^2+0.00001)-ln((x-m_2)^2+0.01)$")
    ax.plot_surface(x, y, z, cmap=cm.coolwarm, edgecolor='royalblue', lw=0.5, alpha=0.5)
    ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max), zlim=(np.min(z), np.max(z)))
    ax.set_xlabel('$x$', fontsize=15)
    ax.set_ylabel('$y$', fontsize=15)
    ax.tick_params(labelsize=12,labelcolor='blue')
    #ax.set_title("f(x,y)")
    if savefig:
        plt.savefig("Rosenbrock.eps",bbox_inches='tight',transparent=True) #bbox_inches='tight', 
    plt.show()


def Ackley(sol):
    """
    This function computes the objective to be minimized.
    It process multiple solutions.
    
    Inputs
    --------
    sol: np.ndarray, each row is a guessed solution (an x-instance).
    dim: int, number of dimensions of the solution space.
    
    Outputs
    --------
    output: np.1darray, the ith entry is the objective value of the ith row of sol.
    """
    dim = sol.shape[1]
    output = 20*np.exp(
                      -0.2*np.sqrt( 0.5*np.sum(sol**2,axis=1) )
                      ) + np.exp(    0.5*np.sum( np.cos(2*np.pi*sol),  axis=1)    )
    return output


def plot_Ackley(savefig=False):
    #Plot the 2D objective
    x_delta = 0.01
    y_delta = 0.01
    x_min = -5
    x_max = 5
    y_min = -5
    y_max = 5
    x = np.arange(x_min,x_max,x_delta)
    y = np.arange(y_min,y_max,y_delta)
    dim = len(x)
    sol = np.zeros((dim**2,2))
    for d in range(dim):
        sol[d*dim:(d+1)*dim,0] = x[d]
        sol[d*dim:(d+1)*dim,1] = y
    
    z = np.reshape(Ackley(sol),(dim,dim))
    ax = plt.figure(figsize=(10,10)).add_subplot(projection='3d')
    x, y = np.meshgrid(x, y)
    # Plot the 3D surface
    #ax.title.set_text(r"$f(x)=-ln((x-m_1)^2+0.00001)-ln((x-m_2)^2+0.01)$")
    ax.plot_surface(x, y, z, cmap=cm.coolwarm, edgecolor='royalblue', lw=0.5, alpha=0.5)
    ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max), zlim=(np.min(z), np.max(z)))
    ax.set_xlabel('$x$', fontsize=15)
    ax.set_ylabel('$y$', fontsize=15)
    ax.tick_params(labelsize=12)
    #ax.set_title("f(x,y)")
    if savefig:
        plt.savefig("Ackley.eps", bbox_inches='tight', transparent=True)
    plt.show()

