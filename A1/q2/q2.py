import numpy as np
import datetime
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_data(Theta: np.ndarray, X: np.ndarray, num_sample, noise = (0,0), save = False) -> np.ndarray:
    '''
        Function to generated data with normal distribution\n
        By default data is generated without noise but noise parameters can be given in input\n
        X | noise: (mean, std_dev)
        Outputs data to X.csv and Y.csv
    '''
    X_data = []
    for i in range(np.shape(X)[0]):
        X_data.append(np.random.normal(X[i,0], X[i,1], num_sample))
    X_data = np.array(X_data)
    gauss_noise = np.random.normal(noise[0], noise[1], num_sample)
    Y_data = np.matmul(Theta.T, X_data) + gauss_noise
    if save:
        np.savetxt('X.csv', X_data.T)
        np.savetxt('Y.csv', Y_data.T)
    return X_data.T, Y_data.T
    
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
    return cost_val

def stochastic_gradient_descent(X: np.ndarray, Y: np.ndarray, Theta: np.ndarray, eta: float, epsilon: float, r: int) -> int:
    '''
        stochastic gradient descent implementation for training set X,Y and parameters Theta with learning rate eta and stopping condition as |delta cost| < epsilon\n
        Returns number of iterations and list of Theta traversed\n
        r is the batch size
    '''
    prev_j = float('inf')
    curr_j = cost(X, Y, Theta)
    iter_ct, epoch_ct = 0, 0
    Theta_list = [np.array(Theta)]
    num_batch = int(np.ceil(np.shape(X)[0]/r))
    while (abs(prev_j-curr_j) > epsilon):       # loop while stopping condition is not satisified
        for batch in range(num_batch):
            X_batch = X[r*batch: r*(batch+1)]
            Y_batch = Y[r*batch: r*(batch+1)]
            # print(np.shape(X_batch))
            Theta -= eta * gradient(X_batch, Y_batch, Theta)
            iter_ct += 1
            Theta_list.append(np.array(Theta))
        epoch_ct += 1
        prev_j = curr_j
        curr_j = cost(X, Y, Theta)
    return iter_ct, epoch_ct, Theta_list

def output_data(file_address: str, Theta: np.ndarray, eta: float, epsilon: float, iter_ct: int, epoch_ct: int, time: int) -> None:
    '''
        Output analyzed model to file
    '''
    file = open(file_address, 'w')
    file.write(f'Learning Rate:\t\t\t\t\t\t\t{eta}\n')
    file.write(f'Stopping Criterion (|\u03B5| > |\u0394 J|):\t\t{epsilon}\n')
    file.write(f'\u03B8:\t\t\t\t\t\t\t\t\t\t{list(Theta.flatten())}\n')
    file.write(f'Number of iterations:\t\t\t\t\t{iter_ct}\n')
    file.write(f'Number of epochs:\t\t\t\t\t\t{epoch_ct}\n')
    file.write(f'Time Taken:\t\t\t\t\t\t\t\t{time} sec')
    file.close()

def read_file(file_address: str) -> np.ndarray:
    '''
        function to read file data and return it as numpy array X, Y
    '''
    try:
        arr = np.loadtxt(file_address, dtype = float, ndmin = 2, skiprows=1, delimiter=',')
        Y = arr[::, 2:]
        X = np.append(np.ones((np.shape(Y)[0],1)), arr[::,:2], axis = 1)
        return X, Y
    except:
        raise Exception('Can not open' + file_address)

def plot_theta(r: int, Theta_list: np.ndarray, file_address: str):
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    ax.set_title(f'Theta Movement for batch size = {r}')
    ax.set_xlabel('$\u03B8_0$')
    ax.set_ylabel('$\u03B8_1$')
    ax.set_zlabel('$\u03B8_2$')
    X = Theta_list[::,0].flatten()
    Y = Theta_list[::,1].flatten()
    Z = Theta_list[::,2].flatten()
    ax.plot(X, Y, Z)
    fig.savefig(file_address)

# Main starts here

# (a) Generating data
Theta = np.array([[3],[1],[2]])
X_dist = np.array([[1,0],[3,2],[-1,2]])
noise_dist = (0, np.sqrt(2))
X, Y = generate_data(Theta, X_dist, 1000000, noise_dist)
print('Data Generated Successfully')

# (c) Calculating error for actual values
X_test, Y_test = read_file('../data/q2/q2test.csv')
print("Data read from q2test.csv succesfully")
error_file = open('(c) error.txt', 'w')
error_file.write('Batch Size\t\terror\n')
err_real = cost(X_test, Y_test, Theta)
error_file.write(f'Original\t\t\t{err_real[0][0]}\n')
print("Mean Square Error for original Theta calculated")

# (b) Running Stochastic Gradient Descent for different epsilon
eta = 0.001
for r, epsilon in [(1, 1e-10), (100, 1e-10), (10000, 1e-10), (1000000, 1e-4)]:
    time_before = datetime.datetime.now()
    Theta_calc = np.array([[0.0],[0.0],[0.0]])
    iter_ct, epoch_ct, Theta_list = stochastic_gradient_descent(X, Y, Theta_calc, eta, epsilon, r)
    time_diff = datetime.datetime.now() - time_before
    output_data(f'(b) Theta r={r} eps = {epsilon}.txt', Theta_calc, eta, epsilon, iter_ct, epoch_ct, time_diff.total_seconds())
    print(f'Stochastic Gradient descent completed for batch size = {r}, epsilon = {epsilon}')

    # (c) Calculating error for actual values
    err = cost(X_test, Y_test, Theta_calc)
    error_file.write(f'r={r}\t\t\t\t{err[0][0]}\n')
    print(f"Mean Square Error for Theta corresponding to r={r} calculated")
    
    # (d) Potting Theta Path
    plot_theta(r, np.array(Theta_list), f'(d) Theta Movement r = {r}.jpg')
    print(f"Graph of Theta movement for Theta corresponding to r={r} plotted")

error_file.close()