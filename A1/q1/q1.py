import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits import mplot3d

def read_file(file_address: str) -> np.ndarray:
    '''
        function to read file data and return it as numpy array
    '''
    try:
        arr = np.loadtxt(file_address, dtype = float, ndmin = 2)
        return arr
    except:
        raise Exception('Can not open' + file_address)

def normalize_data(arr: np.ndarray) -> None:
    '''
        function to normalize a one dimensional numpy array to have zero mean and unit variance
    '''
    arr -= np.mean(arr)         # setting mean to be zero
    arr /= np.std(arr)          # setting unit variance

def gradient(X: np.ndarray, Y: np.ndarray, Theta: np.ndarray) -> np.ndarray:
    '''
        Function to calculate gradient for gradient descent with parameter Theta and training data X,Y
    '''
    grad = np.matmul(X.T, (np.matmul(X,Theta) - Y))/np.shape(X)[0]
    return grad

def cost(X: np.ndarray, Y: np.ndarray, Theta: np.ndarray) -> float:
    '''
        function to calculate the cost function (J(Theta)) for training set X,Y
    '''
    diff = Y - np.matmul(X, Theta)
    cost_val = np.matmul(diff.T, diff)/(2*np.shape(X)[0])
    return cost_val[0][0]

def batch_gradient_descent(X: np.ndarray, Y: np.ndarray, Theta: np.ndarray, eta: float, epsilon: float) -> int:
    '''
        batch gradient descent implementation for training set X,Y and parameters Theta with learning rate eta and stopping condition as |delta cost| < epsilon\n
        Returns number of iterations and list of Theta traversed
    '''
    prev_j = float('inf')
    curr_j = cost(X, Y, Theta)
    iter_ct = 0
    Theta_list = [np.append(Theta, curr_j)]
    while (abs(prev_j-curr_j) > epsilon):       # loop while stopping condition is not satisified
        Theta -= eta * gradient(X, Y, Theta)
        prev_j = curr_j
        curr_j = cost(X, Y, Theta)
        iter_ct += 1
        Theta_list.append(np.append(Theta, curr_j))
    return iter_ct, Theta_list

def output_data(file_address: str, Theta: np.ndarray, eta: float, epsilon: float, iter_ct: int) -> None:
    '''
        Output analyzed model to file
    '''
    file = open(file_address, 'w')
    file.write(f'Learning Rate:\t\t\t\t\t\t\t{eta}\n')
    file.write(f'Stopping Criterion (|\u03B5| > |\u0394 J|):\t\t{epsilon}\n')
    file.write(f'\u03B8:\t\t\t\t\t\t\t\t\t\t{list(Theta.flatten())}\n')
    file.write(f'Number of iterations:\t\t\t\t\t{iter_ct}\n')
    file.close()

def plot_data(X: np.ndarray, Y: np.ndarray, Theta: np.ndarray, title: str, filename: str, Xlabel: str  = 'X', Ylabel: str = 'Y'):
    '''
        function to plot data (in red) as well as hypothesis function (in blue)
    '''
    plt.title(title)
    plt.xlabel(Xlabel)
    plt.ylabel(Ylabel)
    plt.plot(X[::,1:], Y, 'or')        # Removing initial 1s from X and then plotting data
    plt.plot(X[::,1:], np.matmul(X, Theta), '-b')       # Removing initial 1s from X and calculating predicted Y and plotting it
    plt.legend(['Training Data', 'Hypothesis Function'], loc = 'lower right')
    plt.savefig(filename)

def create_mesh(X: np.ndarray, Y: np.ndarray, Theta_list: list):
    '''
        Function to create a 3d mesh of training data
    '''
    theta0 = np.linspace(1.1*Theta_list[0][0]-0.1*Theta_list[-1][0], 2.1*Theta_list[-1][0]-1.1*Theta_list[0][0], 100)
    theta1 = np.linspace(1.1*Theta_list[0][1]-0.1*Theta_list[-1][1], 2.1*Theta_list[-1][1]-1.1*Theta_list[0][1], 100)
    Theta0, Theta1 =  np.array(np.meshgrid(theta0, theta1))
    new_shape = tuple(list(np.shape(Theta0))+[1])                               # new shape for reshaping theta mesh
    Theta_mesh = np.append(Theta0.reshape(new_shape), Theta1.reshape(new_shape), axis = 2)    # reshaping theta mesh to club theta0 and theta1 together
    J = np.apply_along_axis(lambda Theta: cost(X, Y, Theta), 2, Theta_mesh)     # appling cost to all elements of Theta_mesh
    return Theta0, Theta1, J

def plot_mesh(Theta0: np.ndarray ,Theta1: np.ndarray, J: np.ndarray):
    '''
        function to plot a 3d mesh
    '''
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    ax.plot_surface(Theta0, Theta1, J, color='grey', label = 'error funtion mesh', alpha = 0.5)
    ax.set_title('Error Function Mesh')
    ax.set_xlabel('$\u03B8_0$')
    ax.set_ylabel('$\u03B8_1$')
    ax.set_zlabel('J')
    return fig, ax

def animate_mesh(fig, ax, Theta_list: list, animation_address: str, image_address:str):
    '''
        function to animate mesh with each frame showing one step of gradient descent
    '''
    Theta_nplist = np.array(Theta_list)
    line_plot = ax.plot([], [], [], label='Training \u03B8', color='red')[0]
    def update(iter):
        line_plot.set_data(Theta_nplist[:iter+1,:2].T)
        line_plot.set_3d_properties(Theta_nplist[:iter+1,2])
        return line_plot
    anim = animation.FuncAnimation(fig, update, frames = range(len(Theta_list)+1), interval = min(200, 90000/len(Theta_list)))
    anim.save(animation_address, fps = 1000/min(200, 90000/len(Theta_list)))
    line_plot.set_data(Theta_nplist[::,:2].T)
    line_plot.set_3d_properties(Theta_nplist[::,2])
    fig.savefig(image_address)
    
def plot_contour(mesh_gridX: np.ndarray, mesh_gridY: np.ndarray, mesh_gridZ: np.ndarray):
    '''
        function to create contour plots of a mesh grid
    '''
    fig = plt.figure()
    ax = fig.add_subplot()
    contour = ax.contour(meshgridX, meshgridY, meshgridZ, cmap='viridis', levels=30)
    ax.set_title('Error Function Contour Plot')
    ax.set_xlabel('$\u03B8_0$')
    ax.set_ylabel('$\u03B8_1$')
    cbar = fig.colorbar(contour, ax = ax)
    cbar.set_label('J')
    return fig, ax
    
def animate_contour(fig, ax, Theta_list: list, animation_address: str, image_address: str, save: bool):
    '''
        function to animate contour with each frame showing one step of gradient descent
    '''
    Theta_nplist = np.array(Theta_list)
    line_plot = ax.plot([], [], label='Training \u03B8', color='red', marker='x')[0]
    def update(iter):
        line_plot.set_data(Theta_nplist[:iter+1,:2].T)
        return line_plot
    anim = animation.FuncAnimation(fig, update, frames = range(len(Theta_list)+1), interval = min(200, 90000/len(Theta_list)))
    if save:
        anim.save(animation_address, fps = 1000/min(200, 90000/len(Theta_list)))
    else:
        plt.show()
    line_plot.set_data(Theta_nplist[::,:2].T)
    fig.savefig(image_address)


## Main starts here

# Calculating Theta (part a)
arrX = read_file('../data/q1/linearX.csv')
arrY = read_file('../data/q1/linearY.csv')
normalize_data(arrX)
arrX = np.append(np.ones(np.shape(arrX)), arrX, axis = 1)                       # Inserts one to each row of array for constant value of x0
Theta = np.array([[0.0],[0.0]])                                                 # Initialize Theta with zero
eta = 0.03                                                                      # Setting up the learning rate
epsilon = 1e-14                                                                 # Setting up the stopping condition (accuracy) of our search
iter_ct, Theta_list = batch_gradient_descent(arrX, arrY, Theta, eta, epsilon)   # Running batch gradient descent
output_data('(a) Theta.txt', Theta, eta, epsilon, iter_ct)                          # Storing output to file
print('(a)Calculated theta')

# Creating regression plot (part b)
plot_data(arrX, arrY, Theta, 'Linear Regression', '(b) Regression.jpg', 'Normalized Acidity', 'Density of Wine')
print('(b)Plotted data along with hypothesis function')

# Creating Mesh (part c)
meshgridX, meshgridY, meshgridZ = create_mesh(arrX, arrY, Theta_list)
mesh_fig, mesh_axes = plot_mesh(meshgridX, meshgridY, meshgridZ)
animate_mesh(mesh_fig, mesh_axes, Theta_list, '(c) Mesh Animation.mkv', '(c) Mesh image.jpg')
print('(c)mesh completed')
mesh_axes.clear()

# Creating countor plots (part d)
contour_fig, contour_axes = plot_contour(meshgridX, meshgridY, meshgridZ)
animate_contour(contour_fig, contour_axes, Theta_list, '(d) Contour Animation.mkv', '(d) Contour image.jpg', True)
print('(d)contours done')
contour_axes.clear()

# Creating contour plots for different eta (part e)
for eta in [0.001, 0.025, 0.1]:
    Theta = np.array([[0.0],[0.0]])
    iter_ct, Theta_list = batch_gradient_descent(arrX, arrY, Theta, eta, epsilon)
    output_data(f'(e)Calculated Theta (eta = {eta}).txt', Theta, eta, epsilon, iter_ct)
    Theta_list = np.array(Theta_list)
    contour_fig, contour_axes = plot_contour(meshgridX, meshgridY, meshgridZ)
    contour_axes.set_title(f'Error Function Contour Plot\n(\u03B7 = {eta})')
    animate_contour(contour_fig, contour_axes, Theta_list, f'(e) Contour Animation (eta={eta}).mkv', f'(e) Contour image (eta={eta}).jpg', False)
    print(f'(e)contour done for eta={eta}')