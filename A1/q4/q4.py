import numpy as np
import matplotlib.pyplot as plt

def read_file(file_address: str, data_type = float) -> np.ndarray:
    '''
        function to read file data and return it as numpy array
    '''
    try:
        arr = np.loadtxt(file_address, dtype = data_type, ndmin = 2)
        return arr
    except:
        raise Exception('Can not open' + file_address)

def normalize_data(arr: np.ndarray) -> None:
    '''
        function to normalize a one dimensional numpy array to have zero mean and unit variance
    '''
    arr -= np.mean(arr, axis=0)         # setting mean to be zero
    arr /= np.std(arr, axis=0)          # setting unit variance

def GDA(X: np.ndarray, Y: np.ndarray):
    '''
        Function to calculate GDA parameters using closed equation form
    '''
    phi = np.mean(Y)
    mean0 = np.mean(X[np.where(Y<0.5)[0], :], axis = 0)
    mean1 = np.mean(X[np.where(Y>0.5)[0], :], axis = 0)
    X_diff = np.where(Y > 0.5, X-mean1, X-mean0)
    cov = np.matmul(X_diff.T, X_diff)/np.shape(X)[0]
    return phi, mean0, mean1, cov

def general_GDA(X: np.ndarray, Y: np.ndarray):
    '''
        Function to calculate general GDA parameters using closed equation form
    '''
    phi = np.mean(Y)
    mean0 = np.mean(X[np.where(Y<0.5)[0], :], axis = 0)
    mean1 = np.mean(X[np.where(Y>0.5)[0], :], axis = 0)
    ct0 = np.shape(np.where(Y<0.5)[0])[0]
    ct1 = np.shape(np.where(Y>0.5)[0])[0]
    X_diff0 = np.where(Y > 0.5, np.zeros(np.shape(mean0)), X-mean0)
    X_diff1 = np.where(Y < 0.5, np.zeros(np.shape(mean1)), X-mean1)
    cov0 = np.matmul(X_diff0.T, X_diff0)/ct0
    cov1 = np.matmul(X_diff1.T, X_diff1)/ct1
    return phi, mean0, mean1, cov0, cov1

def output_data(file_address: str, phi: float, mean0: np.ndarray, mean1: np.ndarray, cov0: np.ndarray, cov1: np.ndarray) -> None:
    '''
        Output analyzed model to file
    '''
    file = open(file_address, 'w')
    file.write(f'\u03C6:\t\t\t{phi}\n')
    file.write(f'\u03BC0:\t\t\t{mean0.tolist()}\n')
    file.write(f'\u03BC1:\t\t\t{mean1.tolist()}\n')
    if np.array_equal(cov0, cov1):
        file.write(f'\u03A3:\t\t\t{cov0.tolist()}\n')
    else:
        file.write(f'\u03A30:\t\t\t{cov0.tolist()}\n')
        file.write(f'\u03A31:\t\t\t{cov1.tolist()}\n')
    file.close()

def plot_data(X: np.ndarray, Y: np.ndarray) -> tuple:
    '''
        function to plot data of scattered points and boundary separating the region
    '''
    plt.title('Gaussian Discriminant Analysis')
    plt.xlabel('$x_1$ = diameter in fresh water')
    plt.ylabel('$x_2$ = diameter in marine water')
    label0 = np.where(Y[:,0]<=0.5)[0]
    label1 = np.where(Y[:,0]>0.5)[0]
    plt.scatter(X[label0,0], X[label0,1], label='Alaska', marker='o', c = 'b')     # Plotting point with value 0
    plt.scatter(X[label1,0], X[label1,1], label='Canada', marker='x', c = 'r')     # Plotting point with value 1
    # plt.legend()
    return plt.xlim(), plt.ylim()
    
def plot_implicit(data_range: tuple, color: str, label: str, imp_fun) -> None:
    '''
        function to plot any implicit function imp_fun\n
        data_range is (x_range, y_range) for which graph is to be plotted
    '''
    x = np.linspace(data_range[0][0], data_range[0][1], 100)
    y = np.linspace(data_range[1][0], data_range[1][1], 100)
    X, Y = np.meshgrid(x, y)
    vectorize_imp_fun = np.vectorize(imp_fun)
    Z = vectorize_imp_fun(X, Y)
    plt.contour(X, Y, Z, levels=[0], colors = color)
    plt.plot([],[], c = color, label = label)
    
def plot_linear_boundary(phi: float, mean0: np.ndarray, mean1: np.ndarray, cov: np.ndarray, data_range: tuple):
    '''
        function to plot linear boundary between different labels
    '''
    invcov = np.linalg.inv(cov)
    def implicit(x1, x2):
        X0 = np.array([[x1],[x2]]) - mean0
        X1 = np.array([[x1],[x2]]) - mean1
        z = 2*np.log(phi/(1-phi)) + np.matmul(X0.T, np.matmul(invcov, X0)) - np.matmul(X1.T, np.matmul(invcov, X1))
        return z
    plot_implicit(data_range, 'orange', 'linear boundary', implicit)

def plot_quadratic_boundary(phi: float, mean0: np.ndarray, mean1: np.ndarray, cov0: np.ndarray, cov1: np.ndarray, data_range: tuple):
    '''
        function to plot quadratic boundary between different labels
    '''
    invcov0 = np.linalg.inv(cov0)
    invcov1 = np.linalg.inv(cov1)
    def implicit(x1, x2):
        X0 = np.array([[x1],[x2]]) - mean0
        X1 = np.array([[x1],[x2]]) - mean1
        z = 2*np.log(phi/(1-phi)) + np.matmul(X0.T, np.matmul(invcov0, X0)) - np.matmul(X1.T, np.matmul(invcov1, X1))
        return z
    plot_implicit(data_range, 'g', 'quadratic boundary', implicit)


# Main starts here

# (a) Implementing GDA
arrX = read_file('../data/q4/q4x.dat')
normalize_data(arrX)
arrY = read_file('../data/q4/q4y.dat', data_type= str)
arrY = np.where(arrY == 'Alaska', 0.0, 1.0)         # Assigning 0.0 to Alaska and 1.0 to Canada
phi, mean0, mean1, cov = GDA(arrX, arrY)
output_data('(a) GDA.txt', phi, mean0, mean1, cov, cov)

# (b) Making a scatter plot of data
data_range = plot_data(arrX, arrY)

# (c) Plotting linear boundary of separation
mean0 = mean0.reshape((-1,1))
mean1 = mean1.reshape((-1,1))
plot_linear_boundary(phi, mean0, mean1, cov, data_range)

# (d) Implementing General GDA
phi, mean0, mean1, cov0, cov1 = general_GDA(arrX, arrY)
output_data('(d) general GDA.txt', phi, mean0, mean1, cov0, cov1)

# (e) Plotting quadratic boundary of separation
mean0 = mean0.reshape((-1,1))
mean1 = mean1.reshape((-1,1))
plot_quadratic_boundary(phi, mean0, mean1, cov0, cov1, data_range)

# Saving plot
plt.legend()
plt.savefig('(bce) GDA.jpg')