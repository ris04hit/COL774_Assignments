import os
import sys
from PIL import Image
import numpy as np
import pandas as pd
import cvxopt
import libsvm.svmutil
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix
import seaborn as sns

def read_data(folder_path: str, img_size: tuple):        # Reads all the images in folder (assuming only images are present)
    file_list = os.listdir(folder_path)
    image_data = []
    for file in file_list:
        img_address = f'{folder_path}/{file}'
        img = Image.open(img_address)
        resized_img = img.resize(img_size).convert("RGB")
        normalized_img = np.array(resized_img)/255
        flattened_img = normalized_img.reshape(-1)
        image_data.append(flattened_img)
    return np.array(image_data)

def binary_data(image_data1: np.ndarray, image_data2: np.ndarray):  # Combines two image data files with y value of 1 to img_data 1 and -1 to img_data 2
    df1 = pd.DataFrame({colName_image: image_data1.tolist()})
    df2 = pd.DataFrame({colName_image: image_data2.tolist()})
    df1.insert(1, colName_class, np.ones((image_data1.shape[0],), dtype=int))
    df2.insert(1, colName_class, -np.ones((image_data1.shape[0],), dtype=int))
    df = pd.concat([df1, df2], ignore_index=True)
    df[colName_image] = df[colName_image].apply(np.array)
    return df

def linear_kernel(vec1: np.ndarray, vec2: np.ndarray):      # Implements a linear kernel
    return np.matmul(vec1.T, vec2)

def gaussian_kernel(vec1: np.ndarray, vec2: np.ndarray):        # Implements a gaussian kernel
    vec_diff = vec1 - vec2
    return np.exp(-gamma * np.matmul(vec_diff.T, vec_diff))

def store_SVM_param(P: np.ndarray, class_val1: int, class_val2: int, msg: str):       # Stores SVM parameters in a file
    if not os.path.exists('temp'):
        os.makedirs('temp')
    np.save(f'temp/{msg}_P{class_val1}{class_val2}.npy', P)

def load_SVM_param(class_val1: int, class_val2: int, msg: str):       # Loads SVM parameters from file
    return np.load(f'temp/{msg}_P{class_val1}{class_val2}.npy')

def SVM_lin(df: pd.DataFrame, class_val1: int, class_val2: int, msg, read):      # Function for support vector machine given kernel
    start_time = time.time()
    data_size = len(df)
    P = None
    if read:
        try:
            P = cvxopt.matrix(load_SVM_param(class_val1, class_val2, msg))
        except:
            pass
    if not P:
        P = np.array([[df[colName_class][i]*df[colName_class][j]*linear_kernel(df[colName_image][i], df[colName_image][j]) for i in range(data_size)] for j in range(data_size)])
        store_SVM_param(P, class_val1, class_val2, msg)
        P =cvxopt.matrix(P)
    Q = cvxopt.matrix(-np.ones((data_size,), dtype=float))
    G = cvxopt.matrix(np.concatenate([np.identity(data_size), -np.identity(data_size)], axis=0))
    H = cvxopt.matrix(np.concatenate([np.full((data_size,), C), np.zeros((data_size,), dtype=float)], axis=0))
    A = cvxopt.matrix(np.array(df[colName_class].astype(float)).reshape((1, data_size)))
    B = cvxopt.matrix([0.0])
    solution = cvxopt.solvers.qp(P, Q, G, H, A, B)
    alpha = np.array(solution['x']).reshape((-1))
    w = np.matmul(alpha.T, np.array(df[colName_class]*df[colName_image]))
    b = 0
    support_vector_arr = []
    for ind in range(data_size):
        if tolerance < alpha[ind] < C-tolerance:
            b += (df[colName_class][ind] - np.matmul(np.array(df[colName_image][ind]), w))
            support_vector_arr.append(ind)
    b/=len(support_vector_arr)
    end_time = time.time()
    return alpha, w, b, support_vector_arr, end_time - start_time

def SVM_gauss(df: pd.DataFrame, class_val1: int, class_val2: int, msg, read):      # Function for support vector machine given kernel
    start_time = time.time()
    data_size = len(df)
    P = None
    if read:
        try:
            P = cvxopt.matrix(load_SVM_param(class_val1, class_val2, msg))
        except:
            pass
    if not P:
        P = np.array([[df[colName_class][i]*df[colName_class][j]*gaussian_kernel(df[colName_image][i], df[colName_image][j]) for i in range(data_size)] for j in range(data_size)])
        store_SVM_param(P, class_val1, class_val2, msg)
        P =cvxopt.matrix(P)
    Q = cvxopt.matrix(-np.ones((data_size,), dtype=float))
    G = cvxopt.matrix(np.concatenate([np.identity(data_size), -np.identity(data_size)], axis=0))
    H = cvxopt.matrix(np.concatenate([np.full((data_size,), C), np.zeros((data_size,), dtype=float)], axis=0))
    A = cvxopt.matrix(np.array(df[colName_class].astype(float)).reshape((1, data_size)))
    B = cvxopt.matrix([0.0])
    solution = cvxopt.solvers.qp(P, Q, G, H, A, B)
    alpha = np.array(solution['x']).reshape((-1))
    b = 0
    support_vector_arr = []
    for ind in range(data_size):
        if tolerance < alpha[ind] < C-tolerance:
            b += (df[colName_class][ind])
            b -= np.matmul(alpha,np.array([df[colName_class][ind]*P[ind, i] for i in range(data_size)]))
            support_vector_arr.append(ind)
    b/=len(support_vector_arr)
    end_time  =time.time()
    return alpha, b, support_vector_arr, end_time - start_time

def SVM_classifier_linear(img_arr: np.ndarray, w: np.ndarray, b: np.ndarray):     # Classify img_arr
    if np.matmul(w.T, img_arr) + b >= 0:
        return 1
    return -1

def SVM_classifier_gaussian(df: pd.DataFrame, img_arr: np.ndarray, alpha: np.ndarray, b: np.ndarray):     # Classify img_arr
    data_size = len(df)
    if np.matmul(alpha,np.array([df[colName_class][i]*gaussian_kernel(img_arr, df[colName_image][i]) for i in range(data_size)])) + b >= 0:
        return 1
    return -1

def model_accuracy(df: pd.DataFrame, svm_classifier, show = False):      # Calculates model accuracy for given svm function
    correct_prediction = 0
    total_prediction = 0
    for index, row in df.iterrows():
        prediction = svm_classifier(row[colName_image])
        if prediction == row[colName_class]:
            correct_prediction += 1
        total_prediction += 1
        if show:
            print(index, prediction, row[colName_class])
    return correct_prediction/total_prediction

def model_accuracy_train_gauss(df: pd.DataFrame, alpha: np.array, b: np.array, class_val1: int, class_val2: int): # calculate train model accuracy for gaussian kernel
    data_size = len(df)
    P = cvxopt.matrix(load_SVM_param(class_val1, class_val2, "gaussian kernel"))
    correct_prediction = 0
    total_prediction = 0
    for ind in range(data_size):
        prediction = -1
        if np.matmul(alpha,np.array([df[colName_class][ind]*P[i, ind] for i in range(data_size)])) + b >= 0:
            prediction = 1
        if prediction == df[colName_class][ind]:
            correct_prediction += 1
        total_prediction += 1
    return correct_prediction/total_prediction

def top_k_img(alpha: np.ndarray):       # returns array of indices of top k images
    top = [0]*top_coeff_num
    data_size = alpha.shape[0]
    for ind in range(data_size):
        index = ind
        elem = alpha[index]
        for i in range(top_coeff_num):
            if elem > alpha[top[i]]:
                elem = alpha[top[i]]
                index, top[i] = top[i], index
    return top

def plot_img(img_data: np.ndarray, size: tuple, address: str, title: str):     # Plots image
    reshaped_arr = img_data.reshape((size[0], size[1], -1))
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.imshow(reshaped_arr)
    ax.axis('off')
    ax.set_title(title)
    fig.savefig(address)

def libsvm_lin(df_train: pd.DataFrame, df_validate: pd.DataFrame):      # Uses libsvm to solve svm
    start_time = time.time()
    param = libsvm.svmutil.svm_parameter(f'-s 0 -t 0 -c {C}')
    problem = libsvm.svmutil.svm_problem(df_train[colName_class].tolist(), df_train[colName_image].tolist())
    model = libsvm.svmutil.svm_train(problem, param)
    end_time = time.time()
    predicted_label_train, accuracy_train, _ = libsvm.svmutil.svm_predict(df_train[colName_class].to_list(), df_train[colName_image].to_list(), model)
    predicted_label_validate, accuracy_validate, _ = libsvm.svmutil.svm_predict(df_validate[colName_class].to_list(), df_validate[colName_image].to_list(), model)
    support_vectors = np.array(model.get_sv_indices())-1
    alpha_temp = np.array(model.get_sv_coef()).reshape((-1,))
    sv_set = set(support_vectors)
    alpha = []
    ind = 0
    for i in range(len(df_train)):
        if i in sv_set:
            alpha.append(alpha_temp[ind])
            ind += 1
        else:
            alpha.append(0.0)
    alpha = np.array(alpha)
    w = np.matmul(alpha.T, np.array(df_train[colName_class]*df_train[colName_image]))
    b = -model.rho[0]
    return accuracy_train[0], accuracy_validate[0], np.array(support_vectors), np.array(w), b, end_time - start_time

def libsvm_gauss(df_train: pd.DataFrame, df_validate: pd.DataFrame):      # Uses libsvm to solve svm
    start_time = time.time()
    param = libsvm.svmutil.svm_parameter(f'-s 0 -t 2 -c {C} -g {gamma}')
    problem = libsvm.svmutil.svm_problem(df_train[colName_class].tolist(), df_train[colName_image].tolist())
    model = libsvm.svmutil.svm_train(problem, param)
    end_time = time.time()
    predicted_label_train, accuracy_train, _ = libsvm.svmutil.svm_predict(df_train[colName_class].to_list(), df_train[colName_image].to_list(), model)
    predicted_label_validate, accuracy_validate, _ = libsvm.svmutil.svm_predict(df_validate[colName_class].to_list(), df_validate[colName_image].to_list(), model)
    support_vectors = np.array(model.get_sv_indices())-1
    return accuracy_train[0], accuracy_validate[0], np.array(support_vectors), end_time - start_time

def apply_all(df_list: list, func):     # Applies func over all elements of 2 dimension df_list
    ret_list = []
    for i in range(len(df_list)):
        ret_list_temp = []
        for j in range(len(df_list[i])):
            if (i>=j):
                ret_list_temp.append(None)
            else:
                ret_list_temp.append(func(df_list[i][j], i, j))
        ret_list.append(ret_list_temp)
    return ret_list

def multi_class_classifier_gauss(df_train_list, svm_result: list, img_data: np.ndarray):       # Multi class classifier
    def SVM_classifier_gaussian_temp(df: pd.DataFrame, img_arr: np.ndarray, alpha: np.ndarray, b: np.ndarray, i: int, j: int):     # Classify img_arr
        data_size = len(df)
        score = np.matmul(alpha,np.array([df[colName_class][i]*gaussian_kernel(img_arr, df[colName_image][i]) for i in range(data_size)])) + b
        if score >= 0:
            return i, score
        return j, score
    vote_score = [[0, 0] for i in range(class_num)]
    for i in range(class_num):
        for j in range(i+1, class_num):
            ind, score = SVM_classifier_gaussian_temp(df_train_list[i][j], img_data, svm_result[i][j][0], svm_result[i][j][1], i, j)
            vote_score[ind][0] += 1
            vote_score[ind][1] += score
    max_ind = 0
    for i in range(class_num):
        if vote_score[i] > vote_score[max_ind]:
            max_ind = i
    return max_ind
    
def multi_class_gauss_lib(df_train_list: list, df_validate_list: list, data_validate: list, filename: str):     # Using model of LIBSVM solve multiclass svm
    global num_mismatch
    start_time = time.time()
    model_list = []
    data_size_validate = len(data_validate[0])
    for i in range(class_num):
        model_list_temp = []
        for j in range(class_num):
            if i>=j:
                model_list_temp.append(None)
                continue
            param = libsvm.svmutil.svm_parameter(f'-s 0 -t 2 -c {C} -g {gamma} -h 0')
            problem = libsvm.svmutil.svm_problem(df_train_list[i][j][colName_class].to_list(), df_train_list[i][j][colName_image].to_list())
            model = libsvm.svmutil.svm_train(problem, param)
            model_list_temp.append(model)
        model_list.append(model_list_temp)
    prediction = []
    actual_val = []
    total = 0
    correct_prediction = 0
    for class_val in range(class_num):
        ct_misclassified = 0
        for img_data in data_validate[class_val]:
            vote_score = [[0,0] for _ in range(class_num)]
            for i in range(class_num):
                for j in range(i+1, class_num):
                    sys.stdout = open(os.devnull, 'w')
                    predicted_label, decision_values, _ = libsvm.svmutil.svm_predict([0], [img_data], model_list[i][j])
                    sys.stdout = sys.__stdout__
                    if int(predicted_label[0]) == 1:
                        vote_score[i][0] += 1
                        vote_score[i][1] += decision_values[0]
                    else:
                        vote_score[j][0] += 1
                        vote_score[j][1] += decision_values[0]
            max_ind = 0
            for i in range(class_num):
                if vote_score[i] > vote_score[max_ind]:
                    max_ind = i
            if max_ind == class_val:
                correct_prediction += 1
            elif num_mismatch and (ct_misclassified < 2):
                plot_img(img_data, (16, 16), f'Misclassified {num_mismatch-1}', f'Misclassified: Original = {class_val}, Predicted = {max_ind}')
                num_mismatch -= 1
                ct_misclassified += 1
            total += 1
            prediction.append(max_ind)
            actual_val.append(class_val)
    end_time = time.time()
    if filename:
        conf_matrix = confusion_matrix(actual_val, prediction)
        fig = plt.figure(figsize=(8,6))
        axes = plt.subplot()
        labels = ['0', '1', '2', '3', '4', '5']
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax = axes, xticklabels=labels, yticklabels=labels)
        axes.set_xlabel('Predicted Labels')
        axes.set_ylabel('True Labels')
        axes.set_title(filename)
        fig.savefig(filename)
    return correct_prediction/total, end_time - start_time

def multi_class_accuracy(data_list: np.ndarray, classifier, filename: str):        # Multi class model accuracy calculator
    prediction = []
    actual_val = []
    total = 0
    correct_prediction = 0
    for class_val in range(class_num):
        for img_data in data_list[class_val]:
            predicted_value = classifier(img_data)
            if predicted_value == class_val:
                correct_prediction += 1
            total += 1
            prediction.append(predicted_value)
            actual_val.append(class_val)
    conf_matrix = confusion_matrix(actual_val, prediction)
    fig = plt.figure(figsize=(8,6))
    axes = plt.subplot()
    labels = ['0', '1', '2', '3', '4', '5']
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax = axes, xticklabels=labels, yticklabels=labels)
    axes.set_xlabel('Predicted Labels')
    axes.set_ylabel('True Labels')
    axes.set_title(filename)
    fig.savefig(filename)
    return correct_prediction/total

def k_break_data(data_list: np.ndarray):    #Splits data in k parts and create training and validation data
    data_train_break_list = []
    data_validate_break_list = []
    data_size = len(data_list[0])
    break_size = data_size//k_val
    shuffled_indices = np.random.permutation(data_list.shape[1])
    shuffled_data = data_list[:, shuffled_indices, :]
    df_segment = [shuffled_data[:,break_size*i:break_size*(i+1),:] for i in range(k_val)]
    for ind in range(k_val):
        data_validate_break_list.append(df_segment[ind])
        data_train_break_list.append(np.concatenate([df_segment[i] for i in range(k_val) if i != ind], axis=1))
    return data_train_break_list, data_validate_break_list

def k_fold_validation_accuracy(data_train_break_list: list, data_validate_break_list: list, data_validate: list, df_validate_list: list):        # Calculate k fold validation accuracy
    tot_accuracy = 0
    tot_val_accuracy = 0
    for ind in range(k_val):
        df_train_list_break = [[binary_data(data_train_break_list[ind][i], data_train_break_list[ind][j]) for j in range(class_num)] for i in range(class_num)]
        df_validate_list_break = [[binary_data(data_validate_break_list[ind][i], data_validate_break_list[ind][j]) for j in range(class_num)] for i in range(class_num)]
        accuracy, _ = multi_class_gauss_lib(df_train_list_break, df_validate_list_break, data_validate_break_list[ind], '')
        val_accuracy, _ = multi_class_gauss_lib(df_train_list_break, df_validate_list, data_validate, '')
        tot_accuracy += accuracy
        tot_val_accuracy += val_accuracy
    return tot_accuracy/k_val, tot_val_accuracy/k_val

def plot_k_fold(C_list, acc_k_fold_valid, acc_validation):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(np.log10(np.array(C_list)), np.array(acc_k_fold_valid), color = 'r', label = '5 Fold Cross Validation Accuracy')
    ax.plot(np.log10(np.array(C_list)), np.array(acc_validation), color = 'b', label = 'Validation Set Accuracy')
    ax.set_xlabel('log10C')
    ax.set_ylabel(f'Accuracy')
    ax.set_title('k Fold Cross Validation')
    ax.legend()
    fig.savefig('k Fold Cross Validation')

def create_report(filename):        # Creates report
    file = open(filename, 'w')
    file.write(f'Number of Support Vectors (linear):\t\t\t{num_sv_linear}\n')
    file.write(f'Fraction of total data (linear):\t\t\t\t{num_sv_linear/len(df_train)}\n')
    file.write(f'Training Accuraccy SVM (linear):\t\t\t\t{acc_sv_linear_train}\n')
    file.write(f'Validation Accuraccy SVM (linear):\t\t\t{acc_sv_linear_validate}\n\n')
    file.write(f'Value of b (linear):\t\t\t\t\t\t\t\t\t\t{b_lin}\n')
    file.write(f'Number of Support Vectors (gaussian):\t\t{num_sv_gaussian}\n')
    file.write(f'Number of same Support Vectors:\t\t\t\t\t{num_sv_gaussian_match}\n')
    file.write(f'Fraction of total data (gaussian):\t\t\t{num_sv_gaussian/len(df_train)}\n')
    file.write(f'Training Accuraccy SVM (gaussian):\t\t\t{acc_sv_gaussian_train}\n')
    file.write(f'Validation Accuraccy SVM (gaussian):\t\t{acc_sv_gaussian_validate}\n\n')
    file.write(f'Value of b (gaussian):\t\t\t\t\t\t\t\t\t{b_gauss}\n')
    file.write(f'Training Accuracy SVM (linear, LIBSVM):\t{acc_sv_lib_lin_train}\n')
    file.write(f'Validation Accuracy SVM (linear, LIBSVM):\t{acc_sv_lib_lin_validate}\n')
    file.write(f'Training Accuracy SVM (gaussian, LIBSVM):\t{acc_sv_lib_gauss_train}\n')
    file.write(f'Validation Accuracy SVM (gaussian, LIBSVM):\t{acc_sv_lib_gauss_validate}\n')
    file.write(f'Norm of diff of w for both method:\t\t\t{w_comp_liblin_svlin}\n')
    file.write(f'Value of b (LIBSVM linear):\t\t\t\t\t\t\t\t{b_lib_lin}\n\n')
    file.write(f'Support Vector Match (lin, LIBSVM lin):\t{num_svlin_liblin_match}\n')
    file.write(f'Support Vector Match (lin, LIBSVM gauss):\t{num_svlin_libgauss_match}\n')
    file.write(f'Support Vector Match (gauss, LIBSVM lin):\t{num_svgauss_liblin_match}\n')
    file.write(f'Support Vector Match (gauss, LIBSVM gauss):\t{num_svgauss_libgauss_match}\n')
    file.write(f'Support Vector Match (LIB lin, LIB gauss):\t{num_libgauss_liblin_match}\n\n')
    file.write(f'Time for cvxopt linear kernel:\t\t\t\t\t{cost_cvx_lin}\n')
    file.write(f'Time for cvxopt gaussian kernel:\t\t\t\t{cost_cvx_gauss}\n')
    file.write(f'Time for LIBSVM linear kernel:\t\t\t\t\t{cost_lib_lin}\n')
    file.write(f'Time for LIBSVM gaussian kernel:\t\t\t\t{cost_lib_gauss}\n\n')
    file.write(f'Accuracy for Multi Class (CVXOPT):\t\t\t{accuracy_multi_gauss_validate}\n')
    file.write(f'Accuracy for Multi Class (LIBSVM):\t\t\t{accuracy_multi_gauss_validate_lib}\n')
    file.write(f'Time for CVXOPT Multi Class:\t\t\t\t\t\t{cost_multi}\n')
    file.write(f'Time for LIBSVM Multi Class:\t\t\t\t\t\t{cost_multi_lib}\n\n')
    file.write(f'Accuracy for k fold cross validation:\t{acc_k_fold_valid}\n')
    file.close()


# __Main__
class_num = 6
data_train = np.array([read_data(f'../data/train/{class_val}', (16, 16)) for class_val in range(class_num)])
data_validate = np.array([read_data(f'../data/val/{class_val}', (16, 16)) for class_val in range(class_num)])
data_custom = np.array([read_data(f'../data/custom/{class_val}', (16, 16)) for class_val in range(class_num)])
print("Image Data Read")
colName_class = 'label'
colName_image = 'image'
C = 1.0
tolerance = 1e-7
gamma = 0.001
top_coeff_num = 6
num_mismatch = 12
k_val = 5


# Binary Classification
df_train = binary_data(data_train[1], data_train[2])
df_validate = binary_data(data_validate[1], data_validate[2])
df_custom = binary_data(data_custom[1], data_custom[2])


# Part(a)
alpha_lin, w_lin, b_lin, support_vector_arr_lin, cost_cvx_lin = SVM_lin(df_train, 1, 2, 'linear kernel', 1)      # For old parameters
print("Solved SVM for linear kernel")

# Support Vectors
num_sv_linear = len(support_vector_arr_lin)

# Model Accuracy
acc_sv_linear_train = model_accuracy(df_train, lambda image_data: SVM_classifier_linear(image_data, w_lin, b_lin))
print("Training Accuracy calculated for linear kernel")
acc_sv_linear_validate = model_accuracy(df_validate, lambda image_data: SVM_classifier_linear(image_data, w_lin, b_lin))
print("Validation Accuracy calculated for linear kernel")
# acc_sv_linear_custom = model_accuracy(df_custom, lambda image_data: SVM_classifier(image_data, w_lin, b_lin), True)

# Plot
top_arr_lin = top_k_img(alpha_lin)
for ind in range(top_coeff_num):
    plot_img(df_train[colName_image][top_arr_lin[ind]], (16, 16), f'linear svm {ind}', f'linear svm {ind}')
plot_img(w_lin, (16, 16), 'linear svm w', 'linear svm w')
print("Images plotted for linear kernel")


# Part(b)
alpha_gauss, b_gauss, support_vector_arr_gauss, cost_cvx_gauss = SVM_gauss(df_train, 1, 2, 'gaussian kernel', 1)      # For old parameters
print("Solved SVM for gaussian kernel")

# Support Vectors
num_sv_gaussian= len(support_vector_arr_gauss)
num_sv_gaussian_match = len(set(support_vector_arr_gauss).intersection(set(support_vector_arr_lin)))

# Model Accuracy
acc_sv_gaussian_train = model_accuracy_train_gauss(df_train, alpha_gauss, b_gauss, 1, 2)
print("Training Accuracy calculated for gaussian kernel")
acc_sv_gaussian_validate = model_accuracy(df_validate, lambda image_data: SVM_classifier_gaussian(df_train, image_data, alpha_gauss, b_gauss))
print("Validation Accuracy calculated for gaussian kernel")
# acc_sv_gaussian_custom = model_accuracy(df_custom, lambda image_data: SVM_classifier(image_data, w_gauss, b_gauss), True)

# Plot
top_arr_gauss = top_k_img(alpha_gauss)
for ind in range(top_coeff_num):
    plot_img(df_train[colName_image][top_arr_gauss[ind]], (16, 16), f'gaussian svm {ind}', f'gaussian svm {ind}')
print("Images plotted for gaussian kernel")


# Part(c)
# Accuracy
acc_sv_lib_lin_train, acc_sv_lib_lin_validate, support_vector_lib_lin, w_lib_lin, b_lib_lin, cost_lib_lin = libsvm_lin(df_train, df_validate)
w_comp_liblin_svlin = sum([np.matmul((w_lib_lin[i] - w_lin[i]).reshape((1,-1)), (w_lib_lin[i] - w_lin[i]).reshape((-1,1))) for i in range(w_lin.shape[0])])/w_lin.shape[0]
print("Calculated Accuracy using LIBSVM for linear model")
acc_sv_lib_gauss_train, acc_sv_lib_gauss_validate, support_vector_lib_gauss, cost_lib_gauss = libsvm_gauss(df_train, df_validate)
print("Calculated Accuracy using LIBSVM for gaussian model")

# Support Vectors

num_svlin_liblin_match = len(set(support_vector_arr_lin).intersection(set(support_vector_lib_lin)))
num_svlin_libgauss_match = len(set(support_vector_arr_lin).intersection(set(support_vector_lib_gauss)))
num_svgauss_liblin_match = len(set(support_vector_arr_gauss).intersection(set(support_vector_lib_lin)))
num_svgauss_libgauss_match = len(set(support_vector_arr_gauss).intersection(set(support_vector_lib_gauss)))
num_libgauss_liblin_match = len(set(support_vector_lib_gauss).intersection(set(support_vector_lib_lin)))


# Multi Class Image Classification
df_train_list = [[binary_data(data_train[i], data_train[j]) for j in range(class_num)] for i in range(class_num)]
df_validate_list = [[binary_data(data_validate[i], data_validate[j]) for j in range(class_num)] for i in range(class_num)]

# Part (a)
# Train Model
result_svm_gauss = apply_all(df_train_list, lambda df, i, j: SVM_gauss(df, i, j, 'gaussian kernel', 1))
cost_multi = sum([sum([elem[3] for elem in row if elem]) for row in result_svm_gauss])
print("Solved SVM for multi class classifier")

# Accuracy Calculator and Confusion matrix
accuracy_multi_gauss_validate = multi_class_accuracy(data_validate, lambda image_data: multi_class_classifier_gauss(df_train_list, result_svm_gauss, image_data), 'Confusion Matrix CVXOPT validate')
print("Accuracy Calculated for validation data and Confusion Matrix created for multi class classification")


# Part (b)
accuracy_multi_gauss_validate_lib, cost_multi_lib = multi_class_gauss_lib(df_train_list, df_validate_list, data_validate, 'Confusion Matrix LIBSVM validate')
print("Solved, Calculated Accuracy and Created Confusion Matrix for LIBSVM gaussian multiclass classification")


# Part (d)
data_train_break_list, data_validate_break_list = k_break_data(data_train)
acc_k_fold_valid = []
acc_validation = []
C_list = [1e-5, 1e-3, 1.0, 5.0, 10.0]
for C in C_list:
    result = k_fold_validation_accuracy(data_train_break_list, data_validate_break_list, data_validate, df_validate_list)
    acc_k_fold_valid.append(result[0])
    acc_validation.append(result[1])
plot_k_fold(C_list, acc_k_fold_valid, acc_validation)


# Create Report
create_report('report.txt')