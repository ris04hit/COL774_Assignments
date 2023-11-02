import numpy as np 
import sys
import pdb
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report as get_metric
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import matplotlib.pyplot as plt 
from sklearn.preprocessing import OneHotEncoder
import os
from time import time

def get_metric(pred, real):
    pred = pred.astype(float)
    print(f'precision:\t{precision_score(real, pred, average=None)}')
    print(f'accuracy:\t{accuracy_score(real, pred)}')
    print(f'f1:\t\t{f1_score(real, pred, average=None)}')
    print(f'recall:\t\t{recall_score(real, pred, average=None)}')

def get_data(x_path, y_path):
    '''
    Args:
        x_path: path to x file
        y_path: path to y file
    Returns:
        x: np array of [NUM_OF_SAMPLES x n]
        y: np array of [NUM_OF_SAMPLES]
    '''
    x = np.load(x_path)
    y = np.load(y_path)

    y = y.astype('float')
    x = x.astype('float')
    
    #normalize x:
    x = 2*(0.5 - x/255)
    
    return x, y

def sigmoid(arr: np.ndarray):       # Implements sigmoid activation function
    return 1/(1+np.exp(-arr))

def err_sigmoid(arr: np.ndarray, pred_val: np.ndarray = np.array(0)):       # Sigmoid error function
    return arr*(1-arr)

def softmax(arr: np.ndarray):       # Implements softmax activation function
    new_arr = np.exp(arr)
    return new_arr/np.sum(new_arr, axis = 1)[:, None]

def err_softmax(arr: np.ndarray, pred_val: np.ndarray = np.array(0)):       # Softmax error function
    return (pred_val - arr)

def relu(arr: np.ndarray):
    return np.maximum(0, arr)

def err_relu(arr: np.ndarray):
    return np.sign(arr)

def str_to_func(func_name: str):        # Converts string to function
    if func_name == 'sigmoid':
        return sigmoid, err_sigmoid
    elif func_name == 'softmax':
        return softmax, err_softmax
    elif func_name == 'relu':
        return relu, err_relu

def accuracy(Y_true: np.ndarray, Y_predict: np.ndarray):    # Calculates accuracy of prediction
    correct_ct = 0
    total_ct = Y_true.shape[0]
    for ind in range(total_ct):
        if Y_true[ind] == Y_predict[ind]:
            correct_ct+=1
    return correct_ct/total_ct


class Nlayer:
    def __init__(self, activation_func, error_func, layer_size: int, input_size: int) -> None:
        self.layer_size = layer_size
        self.input_size = input_size
        self.activation = activation_func
        self.error_func = error_func
        # self.weight = 3*np.ones((self.layer_size, self.input_size))/input_size
        # self.bias = np.ones((self.layer_size, ))
        self.weight = (np.random.rand(self.layer_size, self.input_size)-0.5)*(20/self.input_size)      # may change initial weights and bias proprotionally
        self.bias = (np.random.rand(self.layer_size)-0.5)*2
        
    def output(self, input_vec: np.ndarray):        # Calculates output of given layer
        net = np.matmul(input_vec, self.weight.T) + self.bias
        return self.activation(net)

    def del_error(self, output_vec: np.ndarray, next_layer_error: np.ndarray, y_pred: np.ndarray):      # Calculated error term for current layer
        return self.error_func(output_vec, y_pred)*next_layer_error

    def print_layer(self):      # Prints Layer
        print(f'layer size:\t{self.layer_size}')
        print(f'input size:\t{self.input_size}')
        print(f'weight:\t\t{self.weight}')
        print(f'bias:\t\t{self.bias}')

    def save(self, path: str):
        np.save(path+'/weight', self.weight)
        np.save(path+'/bias', self.bias)
        
    def load(self, path: str):
        self.weight = np.load(path+'/weight.npy')
        self.bias = np.load(path+'/bias.npy')

class NeuralNetwork:
    def __init__(self, hidden_layer: list, input_size: int, output_size: int, hidden_activation: str, output_activation: str) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        hidden_activation_f, hidden_error_f = str_to_func(hidden_activation)
        output_activation_f, output_error_f = str_to_func(output_activation)        # output layer error function is absoulute, does not depend on other layers
        self.output_activation_f = output_activation_f
        self.output_error_f = output_error_f
        # Initializing layers of Neural Network
        self.layers = [Nlayer(hidden_activation_f, hidden_error_f, hidden_layer[0], self.input_size)]
        for ind in range(len(hidden_layer)-1):
            self.layers.append(Nlayer(hidden_activation_f, hidden_error_f, hidden_layer[ind+1], hidden_layer[ind]))
        self.layers.append(Nlayer(lambda x: x, lambda x, y: 1, self.output_size, hidden_layer[len(hidden_layer)-1]))
    
    def cost(self, X: np.ndarray, Y: np.ndarray):       # Calculates cost
        def cost_one(out, y):
            ind = np.argmax(y)
            return -np.log(out[ind])
        return sum([cost_one(out, y) for out, y in zip(self.output(X), Y)])/(X.shape[0])
    
    def output(self, X_data: np.ndarray):      # Calculates output for given X_data
        input_val = X_data
        for layer in self.layers:
            input_val = layer.output(input_val)
        input_val = self.output_activation_f(input_val)
        return input_val

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray, batch_size: int, learning_rate, max_epoch: int = 200, stopping_cond: float = 1e-8, save_str: str = ''):        # Trains Neural Networks Model
        try:
            self.load(batch_size, max_epoch, stopping_cond, save_str)
        except:
            X_data = X_train.copy()
            Y_data = Y_train.copy()
            num_epoch = 0
            num_iter = 0
            total_sample = X_data.shape[0]
            layer_num = len(self.layers)
            prev_err = float('inf')
            curr_err = self.cost(X_data, Y_data)
            while (num_epoch < max_epoch) and (abs(curr_err-prev_err)>stopping_cond):         # Iteration of each epoch
                # Shuffling of data
                perm = np.random.permutation(total_sample)
                X_data = X_data[perm]
                Y_data = Y_data[perm]
                num_epoch += 1
                for batch_ind in range(0, total_sample, batch_size):
                    delta_err = [np.array(0) for layer in self.layers]
                    output_list = []
                    input_list = []
                    num_iter += 1
                    # Forward Propogation
                    input_val = X_data[batch_ind: batch_ind + batch_size]
                    curr_batch_size = input_val.shape[0]
                    for ind in range(layer_num):
                        input_list.append(input_val)
                        output_val = self.layers[ind].output(input_val)
                        output_list.append(output_val)
                        input_val = output_val
                    output_val = self.output_activation_f(output_val)
                    output_list.append(output_val)
                    next_layer_error = np.ones((curr_batch_size, self.output_size))
                    # Backward Propogation
                    for ind in range(layer_num-1, -1, -1):
                        delta_err[ind] = (self.layers[ind].del_error(output_list[ind], next_layer_error, Y_data[batch_ind: batch_ind + batch_size]))
                        if ind == layer_num-1:
                            delta_err[ind] *= self.output_error_f(output_list[-1], Y_data[batch_ind: batch_ind + batch_size])
                        next_layer_error = np.matmul(delta_err[ind], self.layers[ind].weight)
                    # Updating delta weight values
                    for ind in range(layer_num):
                        self.layers[ind].weight += learning_rate(num_epoch)*np.matmul(delta_err[ind].T, input_list[ind])/curr_batch_size
                        self.layers[ind].bias += learning_rate(num_epoch)*np.sum(delta_err[ind], axis = 0)/curr_batch_size
                print(num_epoch, curr_err, accuracy(Y_test, predict(self.output(X_test))))
                prev_err = curr_err
                curr_err = self.cost(X_data, Y_data)
                if abs(curr_err - prev_err) < stopping_cond:
                    break
            self.save(batch_size, max_epoch, stopping_cond, save_str)

    def print_nn(self):
        for layer in self.layers:
            layer.print_layer()
            print()

    def save(self, batch_size: int, max_epoch: int, stopping_cond: float, save_str: str):
        foldername = f"save/nn{save_str}_{self.input_size}_{self.output_size}_{[layer.layer_size for layer in self.layers]}_{batch_size}_{max_epoch}_{stopping_cond}_{self.hidden_activation}_{self.output_activation}"
        os.mkdir(foldername)
        ind = 0
        for layer in self.layers:
            os.mkdir(foldername+f'/{ind}')
            layer.save(foldername+f'/{ind}')
            ind += 1
            
    def load(self, batch_size: int, max_epoch: int, stopping_cond: float, save_str: str):
        foldername = f"save/nn{save_str}_{self.input_size}_{self.output_size}_{[layer.layer_size for layer in self.layers]}_{batch_size}_{max_epoch}_{stopping_cond}_{self.hidden_activation}_{self.output_activation}"
        ind = 0
        for layer in self.layers:
            layer.load(foldername+f'/{ind}')
            ind += 1
    
def predict(output_data: np.ndarray):       # Make predictions based neural network output
    return np.apply_along_axis(np.argmax, axis = 1, arr = output_data) + 1

# Reading Data
X_train, Y_train = get_data('../data/q2/x_train.npy', '../data/q2/y_train.npy')
X_test, Y_test = get_data('../data/q2/x_test.npy', '../data/q2/y_test.npy')

label_encoder = OneHotEncoder(sparse_output = False)
label_encoder.fit(np.expand_dims(Y_train, axis = -1))

Y_train_onehot = label_encoder.transform(np.expand_dims(Y_train, axis = -1))
Y_test_onehot = label_encoder.transform(np.expand_dims(Y_test, axis = -1))

print("Data Read")

# Part (b)
HLU = [1, 5, 10, 50, 100]
M = 32
n = 1024
r = 5
for hlu in HLU:
    nn = NeuralNetwork([hlu], n, r, 'sigmoid', 'softmax')
    nn.fit(X_train, Y_train_onehot, M, lambda x: 0.01)
    print(get_metric(predict(nn.output(X_train)), Y_train))
    print(get_metric(predict(nn.output(X_test)), Y_test))
    
# Part (c)
HL = [[512], [512, 256], [512, 256, 128], [512, 256, 128, 64]]
for hl in HL:
    nn = NeuralNetwork(hl, n, r, 'sigmoid', 'softmax')
    init_time = time()
    nn.fit(X_train, Y_train_onehot, M, lambda x: 0.01, max_epoch=400)
    final_time = time()
    if final_time - init_time >= 5: 
        with open('time.txt', 'a') as f:
            f.write(f'(c) time = {final_time - init_time}\t{hl}')
    print(f'time = {final_time - init_time}')
    print(get_metric(predict(nn.output(X_train)), Y_train))
    print(get_metric(predict(nn.output(X_test)), Y_test))

# Part (d)
HL = [[512], [512, 256], [512, 256, 128], [512, 256, 128, 64]]
for hl in HL:
    nn = NeuralNetwork(hl, n, r, 'sigmoid', 'softmax')
    init_time = time()
    nn.fit(X_train, Y_train_onehot, M, lambda x: 0.01/np.sqrt(x), save_str='activate', max_epoch=400)
    final_time = time()
    if final_time - init_time >= 5: 
        with open('time.txt', 'a') as f:
            f.write(f'(d) time = {final_time - init_time}\t{hl}')
    print(f'time = {final_time - init_time}')
    print(get_metric(predict(nn.output(X_train)), Y_train))
    print(get_metric(predict(nn.output(X_test)), Y_test))

# Part (e)
HL = [[512], [512, 256], [512, 256, 128], [512, 256, 128, 64]]
for hl in HL:
    nn = NeuralNetwork(hl, n, r, 'sigmoid', 'relu')
    init_time = time()
    nn.fit(X_train, Y_train_onehot, M, lambda x: 0.01/np.sqrt(x), save_str='activate', max_epoch=400)
    final_time = time()
    if final_time - init_time >= 5: 
        with open('time.txt', 'a') as f:
            f.write(f'(e) time = {final_time - init_time}\t{hl}')
    print(f'time = {final_time - init_time}')
    print(get_metric(predict(nn.output(X_train)), Y_train))
    print(get_metric(predict(nn.output(X_test)), Y_test))

# Part (f)
HL = [[512], [512, 256], [512, 256, 128], [512, 256, 128, 64]]
for hl in HL:
    mlp = MLPClassifier(hidden_layer_sizes=hl, activation='relu', solver='sgd', alpha=0, batch_size=32, learning_rate='invscaling')
    mlp.fit(X_train, Y_train_onehot)
    Y_pred_train = (mlp.predict(X_train))
    Y_pred_test = (mlp.predict(X_test))
    print(hl)
    get_metric(Y_pred_train, Y_train_onehot)
    get_metric(Y_pred_test, Y_test_onehot)
