import numpy as np
from matplotlib import pyplot as plt

def read_file(file_address: str) -> np.ndarray:
    '''
        function to read file data and return it as numpy array
    '''
    try:
        arr = np.loadtxt(file_address, dtype = float, ndmin = 2, delimiter=',')
        return arr
    except:
        raise Exception('Can not open' + file_address)

def normalize_data(arr: np.ndarray) -> None:
    '''
        function to normalize a one dimensional numpy array to have zero mean and unit variance
    '''
    arr -= np.mean(arr, axis = 0)         # setting mean to be zero
    arr /= np.std(arr, axis = 0)          # setting unit variance

def h_theta(Theta: np.ndarray, X: np.ndarray) -> np.ndarray:
    '''
        function to calculate h_theta (probability of 1 given x for parameter Theta)\n
        here logistic function is sigmoid g(z) = 1/(1+e^-z)
    '''
    Z = np.matmul(X, Theta)
    HT = 1/(1+np.exp(-Z))
    return HT

def gradient(Theta: np.ndarray, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    '''
        function to calculate gradient of log likelihood function for parameter Theta and training data X and Y.\n
        grad L(theta) = (Y-HT)^T.X
    '''
    grad = np.matmul(X.T, (Y-h_theta(Theta, X)))
    return grad

def hessian(Theta: np.ndarray, X: np.ndarray) -> np.ndarray:
    '''
        function to calculate the hessian matrix for Theta and X
    '''
    h_t = h_theta(Theta, X)
    h = h_t * (1-h_t)
    H_T = np.diag(h.reshape((-1,)))
    hess = -np.matmul(X.T, np.matmul(H_T, X))
    return hess

def cost(Theta: np.ndarray, X: np.ndarray, Y: np.ndarray) -> float:
    '''
        Function for finding cost of given Theta for training data X and Y.
    '''
    h_t = h_theta(Theta, X)
    L1 = np.matmul(Y.T, np.log(h_t))
    L2 = np.matmul(1-(Y.T), np.log(1-h_t))
    return (L1+L2)[0][0]

def newton_method(Theta: np.ndarray, X: np.ndarray, Y: np.ndarray, epsilon: float) -> int:
    '''
        function to implement newtons method for parameter Theta, training data X and Y\n
        stopping condition |cost_curr-cost_prev|<epsilon.
    '''
    iter_ct = 0
    cost_prev = float('inf')
    cost_curr = cost(Theta, X, Y)
    while abs(cost_curr-cost_prev)>epsilon:
        H = hessian(Theta, X)
        grad = gradient(Theta, X, Y)
        diff = np.matmul(np.linalg.inv(H), grad)
        Theta -= diff
        iter_ct += 1
        cost_prev = cost_curr
        cost_curr = cost(Theta, X, Y)
    return iter_ct

def output_data(file_address: str, Theta: np.ndarray, epsilon: float, iter_ct: int) -> None:
    '''
        Output analyzed model to file
    '''
    file = open(file_address, 'w')
    file.write(f'Stopping Criterion (|\u03B5| > |\u0394 cost|):\t\t{epsilon}\n')
    file.write(f'\u03B8:\t\t\t\t\t\t\t\t\t\t{list(Theta.flatten())}\n')
    file.write(f'Number of iterations:\t\t\t\t\t{iter_ct}\n')
    file.close()

def plot_data(X: np.ndarray, Y: np.ndarray, Theta: np.ndarray, filename: str):
    '''
        function to plot data of scattered points and boundary separating the region
    '''
    plt.title('Logistic Regression')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    label0 = np.where(Y[:,0]<=0.5)[0]
    label1 = np.where(Y[:,0]>0.5)[0]
    plt.scatter(X[label0,1], X[label0,2], label='value 0', marker='o', c = 'b')     # Plotting point with value 0
    plt.scatter(X[label1,1], X[label1,2], label='value 1', marker='x', c = 'r')     # Plotting point with value 1
    x_left, x_right = plt.xlim()
    x_values = np.linspace(x_left, x_right, 100)                                    # Creating x values for boundary
    y_values = -(Theta[0][0] + Theta[1][0]*x_values)/(Theta[2][0])                  # Calculating y values for boundary
    plt.plot(x_values, y_values, label='boundary', c='g')
    plt.legend()
    plt.savefig(filename)
    

# Main starts here

# (a) Implementing Newtons method
arrX = read_file('../data/q3/logisticX.csv')
arrY = read_file('../data/q3/logisticY.csv')
normalize_data(arrX)
arrX = np.append(np.ones(np.shape(arrY)), arrX, axis = 1)                       # Inserts one to each row of array for constant value of x0
Theta = np.array([[0.0], [0.0], [0.0]])
epsilon = 1e-20
iter_ct = newton_method(Theta, arrX, arrY, epsilon)
output_data('(a) Theta.txt', Theta, epsilon, iter_ct)

# (b) Plotting data and decision boundary
plot_data(arrX, arrY, Theta, '(b) logistic regression.jpg')